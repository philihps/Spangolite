#!/usr/bin/env julia

# include functions
include("utilities.jl")

function computeExpectationValues(modelDict::Dict, tensorUpdateFlag::String)

    # get model properties
    latticeName = modelDict["latticeName"]
    modelName = modelDict["modelName"]
    physicalSpin = modelDict["physicalSpin"]
    setSym = modelDict["setSym"]
    Lx = Int(modelDict["Lx"])
    Ly = Int(modelDict["Ly"])
    unitCell = modelDict["unitCell"]
    modelParameters = modelDict["modelParameters"]
    chiB = Int(modelDict["chiB"])
    chiE = Int(modelDict["chiE"])

    # get size of unitCell (necessary for e.g. Lx = 1, Ly = 2)
    unitCellLx, unitCellLy = size(unitCell)

    # get number of tensors in the unitCell
    uniqueTensors = unique(unitCell)
    numberOfUniqueTensors = length(uniqueTensors)

    # set physical system
    if setSym == "SU0"
        vecSpacePhys = ComplexSpace(Integer(2 * physicalSpin + 1))
    end

    # define number of simplex, gamma and lambda tensors per unit cell
    if tensorUpdateFlag == "iPESS" || tensorUpdateFlag == "iPESO"
        numSimTensors = 2
        numGamTensors = 3
        numLamTensors = 3 * numSimTensors
    end

    # construct fusing isomorphism
    fuseIsometry = TensorKit.isomorphism(fuse(vecSpacePhys, fuse(vecSpacePhys, fuse(vecSpacePhys, fuse(vecSpacePhys, fuse(vecSpacePhys, vecSpacePhys))))), vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys)

    # constuct nearest-neighbour spin-spin interaction
    if setSym == "SU0"
        Sx, Sy, Sz, Sm, Sp, Id = getSpinOperators(physicalSpin)
        twoBodyGateSpinSpin = (Sx ⊗ Sx + Sy ⊗ Sy + Sz ⊗ Sz)
    end

    # set main directory path
    mainDirPath = @sprintf("numFiles_%s/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", tensorUpdateFlag, latticeName, modelName, physicalSpin, setSym, Lx, Ly)

    # construct folderStringExpectValues
    folderStringExpectValues = mainDirPath * "/expectationValues/bondDim_" * string(chiB)
    ~isdir(folderStringExpectValues) && mkpath(folderStringExpectValues)

    # construct fileStrings
    fileStringExpectValues = @sprintf("%s/expVals_hamVars_", folderStringExpectValues)
    for varIdx = eachindex(modelParameters)
        if varIdx < length(modelParameters)
            fileStringExpectValues *= @sprintf("%+0.2f_", modelParameters[varIdx])
        else
            fileStringExpectValues *= @sprintf("%+0.3f_", modelParameters[varIdx])
        end
    end
    fileStringExpectValues *= @sprintf("chiB_%d_chiE_%d.jld", chiB, chiE)


    #----------------------------------------------------------------------
    # load simulation files
    #----------------------------------------------------------------------

    # construct folderString
    folderStringEnvironments = mainDirPath * "/environmentTensors/bondDim_" * string(chiB)

    # construct fileStrings
    fileStringEnvironments = @sprintf("%s/CTM_hamVars_", folderStringEnvironments)
    for varIdx = eachindex(modelParameters)
        if varIdx < length(modelParameters)
            fileStringEnvironments *= @sprintf("%+0.2f_", modelParameters[varIdx])
        else
            fileStringEnvironments *= @sprintf("%+0.3f_", modelParameters[varIdx])
        end
    end
    fileStringEnvironments *= @sprintf("chiB_%d_chiE_%d.jld", chiB, chiE)

    # load coarseGrainedTensorRegularDicts, coarseGrainedTensorCombineDicts and environmentTensorDicts
    coarseGrainedTensorRegularDicts, coarseGrainedTensorCombineDicts, environmentTensorDicts = load(fileStringEnvironments, "coarseGrainedTensorRegularDicts", "coarseGrainedTensorCombineDicts", "environmentTensorDicts")

    # convert coarseGrainedTensorRegularDicts and coarseGrainedTensorCombineDicts to TensorMaps
    coarseGrainedTensorsRegular = convert.(TensorMap, coarseGrainedTensorRegularDicts)
    coarseGrainedTensorsCombine = convert.(TensorMap, coarseGrainedTensorCombineDicts)

    # convert environmentTensorDicts to environmentTensors
    environmentTensors = Vector{Vector{TensorMap}}(undef, numberOfUniqueTensors)
    for (idx, envTensorDicts) in enumerate(environmentTensorDicts)
        environmentTensors[idx] = convert.(TensorMap, envTensorDicts)
    end

    # set number of lattice sites per unit cell
    numLatticeSitesPerUC = 6

    # initialize array to spin-spin correlation and bond energies
    listOfBondExpVals = zeros(Float64, 0, 4)

    # initialize array to store ground state magnetization
    listOfMagnetizations = zeros(Float64, 0, 4)

    # loop over all unit cells on the Square-Kagome lattice
    for idx = 1 : Lx, idy = 1 : Ly

        if tensorUpdateFlag == "iPESS"


            #----------------------------------------------------------------
            # inter-unit-cell terms for simplex U
            #----------------------------------------------------------------

            # select coarse-grained tensors and environment tensors
            tensorNum = getUnitCellNumber(idx, idy, unitCell)
            gammaTensor = coarseGrainedTensorsRegular[tensorNum]
            envTensors = environmentTensors[tensorNum]

            # compute one-site norm
            oneSiteRDM = computeOneSiteRDM_mapleLeaf(gammaTensor, envTensors)
            oneSiteNorm = real(tr(oneSiteRDM))

            # bond 1-2 mod UC
            gammaNums = [1 2] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[-1, -2, 3, 4, 5, 6, -3, -4, 3, 4, 5, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))

            # bond 1-5 mod UC
            gammaNums = [1 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[-1, 2, 3, 4, -2, 6, -3, 2, 3, 4, -4, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

            # bond 2-3 mod UC
            gammaNums = [2 3] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, -2, 4, 5, 6, 1, -3, -4, 4, 5, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

            # bond 2-4 mod UC
            gammaNums = [2 4] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, 3, -2, 5, 6, 1, -3, 3, -4, 5, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

            # bond 2-5 mod UC
            gammaNums = [2 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, 3, 4, -2, 6, 1, -3, 3, 4, -4, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

            # bond 3-4 mod UC
            gammaNums = [3 4] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, -1, -2, 5, 6, 1, 2, -3, -4, 5, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))

            # bond 4-5 mod UC
            gammaNums = [4 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, -1, -2, 6, 1, 2, 3, -3, -4, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

            # bond 4-6 mod UC
            gammaNums = [4 6] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, -1, 5, -2, 1, 2, 3, -3, 5, -4]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

            # bond 5-6 mod UC
            gammaNums = [5 6] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, 4, -1, -2, 1, 2, 3, 4, -3, -4]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))


            #----------------------------------------------------------------
            # intra-unit-cell terms for simplex D
            #----------------------------------------------------------------

            # construct four-site network for three-site expectation values
            selectBulkPEPOs = [unitCell[pIndex(idxx, unitCellLx), pIndex(idyy, unitCellLy)] for idxx = [idx, idx + 1], idyy = [idy, idy + 1]]
            bulkPEPOs = coarseGrainedTensorsRegular[selectBulkPEPOs]
            bulkPEPOEnvironments = environmentTensors[selectBulkPEPOs]

            # select tensor numbers for position 1, 2, ..., 6
            tensorNumbers = [
                getTensorNumber(idx + 1, idy + 1, unitCell, numLatticeSitesPerUC) + 1, 
                getTensorNumber(idx + 1, idy + 1, unitCell, numLatticeSitesPerUC) + 2, 
                getTensorNumber(idx + 0, idy + 0, unitCell, numLatticeSitesPerUC) + 3, 
                getTensorNumber(idx + 0, idy + 0, unitCell, numLatticeSitesPerUC) + 4, 
                getTensorNumber(idx + 1, idy + 0, unitCell, numLatticeSitesPerUC) + 5, 
                getTensorNumber(idx + 1, idy + 0, unitCell, numLatticeSitesPerUC) + 6
            ]

            # compute three-site reduced density matrix for sites the gates acts on
            threeSiteRDM = computeThreeSiteRDM_mapleLeaf(bulkPEPOs, bulkPEPOEnvironments)
            threeSiteNorm = real(tr(threeSiteRDM))
            
            # bond 1-3 mod UC
            gammaNums = [tensorNumbers[1] tensorNumbers[3]]
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, -2, 4, 5, 6, -3, 2, -4, 4, 5, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))

            # bond 1-4 mod UC
            gammaNums = [tensorNumbers[1] tensorNumbers[4]]
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, 3, -2, 5, 6, -3, 2, 3, -4, 5, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

            # bond 1-6 mod UC
            gammaNums = [tensorNumbers[1] tensorNumbers[6]]
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, 3, 4, 5, -2, -3, 2, 3, 4, 5, -4]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))

            # bond 2-6 mod UC
            gammaNums = [tensorNumbers[2] tensorNumbers[6]]
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, -1, 3, 4, 5, -2, 1, -3, 3, 4, 5, -4]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

            # bond 3-5 mod UC
            gammaNums = [tensorNumbers[3] tensorNumbers[5]]
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, 2, -1, 4, -2, 6, 1, 2, -3, 4, -4, 6]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

            # bond 3-6 mod UC
            gammaNums = [tensorNumbers[3] tensorNumbers[6]]
            @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, 2, -1, 4, 5, -2, 1, 2, -3, 4, 5, -4]
            twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
            listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))
            
        end


        #----------------------------------------------------------------
        # compute magnetizations for SU0
        #----------------------------------------------------------------

        if setSym == "SU0"

            # select coarse-grained tensors and environment tensors
            gammaTensor = coarseGrainedTensorsCombine[idx, idy]
            envTensors = environmentTensors[idx, idy]

            # compute one-site norm
            oneSiteRDM = computeOneSiteRDM_iPEPS(gammaTensor, envTensors)
            oneSiteNorm = real(tr(oneSiteRDM))


            #----------------------------------------------------------------
            # inter-unit-cell terms
            #----------------------------------------------------------------

            # tensor 1 mod UC
            gammaNum = 1 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            expValsMag = zeros(Float64, 0)
            for localOperator = [Sx, Sy, Sz]
                @tensor oneSiteOperator[-1; -2] := fuseIsometry[-1, 1, 2, 3, 4, 5, 6] * localOperator[1, 7] * fuseIsometry'[7, 2, 3, 4, 5, 6, -2]
                oneSiteGate = tr(oneSiteOperator * oneSiteRDM)
                expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
            end
            listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

            # tensor 2 mod UC
            gammaNum = 2 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            expValsMag = zeros(Float64, 0)
            for localOperator = [Sx, Sy, Sz]
                @tensor oneSiteOperator[-1; -2] := fuseIsometry[-1, 1, 2, 3, 4, 5, 6] * localOperator[2, 7] * fuseIsometry'[1, 7, 3, 4, 5, 6, -2]
                oneSiteGate = tr(oneSiteOperator * oneSiteRDM)
                expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
            end
            listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

            # tensor 3 mod UC
            gammaNum = 3 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            expValsMag = zeros(Float64, 0)
            for localOperator = [Sx, Sy, Sz]
                @tensor oneSiteOperator[-1; -2] := fuseIsometry[-1, 1, 2, 3, 4, 5, 6] * localOperator[3, 7] * fuseIsometry'[1, 2, 7, 4, 5, 6, -2]
                oneSiteGate = tr(oneSiteOperator * oneSiteRDM)
                expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
            end
            listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

            # tensor 4 mod UC
            gammaNum = 4 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            expValsMag = zeros(Float64, 0)
            for localOperator = [Sx, Sy, Sz]
                @tensor oneSiteOperator[-1; -2] := fuseIsometry[-1, 1, 2, 3, 4, 5, 6] * localOperator[4, 7] * fuseIsometry'[1, 2, 3, 7, 5, 6, -2]
                oneSiteGate = tr(oneSiteOperator * oneSiteRDM)
                expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
            end
            listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

            # tensor 5 mod UC
            gammaNum = 5 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            expValsMag = zeros(Float64, 0)
            for localOperator = [Sx, Sy, Sz]
                @tensor oneSiteOperator[-1; -2] := fuseIsometry[-1, 1, 2, 3, 4, 5, 6] * localOperator[5, 7] * fuseIsometry'[1, 2, 3, 4, 7, 6, -2]
                oneSiteGate = tr(oneSiteOperator * oneSiteRDM)
                expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
            end
            listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

            # tensor 6 mod UC
            gammaNum = 6 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
            expValsMag = zeros(Float64, 0)
            for localOperator = [Sx, Sy, Sz]
                @tensor oneSiteOperator[-1; -2] := fuseIsometry[-1, 1, 2, 3, 4, 5, 6] * localOperator[6, 7] * fuseIsometry'[1, 2, 3, 4, 5, 7, -2]
                oneSiteGate = tr(oneSiteOperator * oneSiteRDM)
                expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
            end
            listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

        end

    end

    # sort listOfMagnetizations according to lattice sites
    listOfMagnetizations = sortslices(listOfMagnetizations, dims = 1)


    #----------------------------------------------------------------
    # compute bond energies
    #----------------------------------------------------------------

    bondEnergies = zeros(Float64, size(listOfBondExpVals, 1))
    disconCorrel = zeros(Float64, size(listOfBondExpVals, 1))
    for bondIdx = axes(listOfBondExpVals, 1)
        bondInfo = convert.(Int, listOfBondExpVals[bondIdx, 1 : 2])
        expValSzA = listOfMagnetizations[bondInfo[1], 4]
        expValSzB = listOfMagnetizations[bondInfo[2], 4]
        bondEnergies[bondIdx] = prod(listOfBondExpVals[bondIdx, 3 : 4]) - 1/5 * modelParameters[6] * (expValSzA + expValSzB)
        disconCorrel[bondIdx] = listOfBondExpVals[bondIdx, 3] - (expValSzA * expValSzB)
    end
    listOfBondExpVals = hcat(listOfBondExpVals, bondEnergies, disconCorrel)
    
    display(listOfBondExpVals)
    averageBondEnergy = 15/6 * sum(listOfBondExpVals[:, 5]) / size(listOfBondExpVals, 1)
    println(averageBondEnergy, "\n")
    
    display(listOfMagnetizations)
    magPerSite = sqrt.(listOfMagnetizations[:, 2] .^ 2 + listOfMagnetizations[:, 3] .^ 2 + listOfMagnetizations[:, 4] .^ 2)
    averageMagPerSite = 1 / length(magPerSite) * sum(magPerSite.^2)
    averageMagZ = 1 / size(listOfMagnetizations, 1) * sum(listOfMagnetizations[:, 4]) / physicalSpin
    println(averageMagPerSite)
    println(averageMagZ)
    println("\n")

    # store listOfBondExpVals, listOfBondVariances and listOfMagnetizations
    save(fileStringExpectValues, "listOfBondExpVals", listOfBondExpVals, "listOfMagnetizations", listOfMagnetizations)
    println(fileStringExpectValues)

end

function computeGroundStateEnergy_MF(modelDict::Dict, tensorUpdateFlag::String)

    # get model properties
    latticeName = modelDict["latticeName"]
    modelName = modelDict["modelName"]
    physicalSpin = modelDict["physicalSpin"]
    setSym = modelDict["setSym"]
    Lx = Int(modelDict["Lx"])
    Ly = Int(modelDict["Ly"])
    unitCell = modelDict["unitCell"]
    modelParameters = modelDict["modelParameters"]
    chiB = Int(modelDict["chiB"])

    if tensorUpdateFlag == "iPESO"
        β = modelDict["betaT"]
        δβ = modelDict["dBeta"]
    end

    # get size of unitCell (necessary for e.g. Lx = 1, Ly = 2)
    unitCellLx, unitCellLy = size(unitCell)

    # set physical system
    if setSym == "SU0"
        vecSpacePhys = ComplexSpace(Integer(2 * physicalSpin + 1))
    end

    # define number of simplex, gamma and lambda tensors per unit cell
    if tensorUpdateFlag == "iPESS" || tensorUpdateFlag == "iPESO"
        numSimTensors = 2
        numGamTensors = 3
        numLamTensors = 3 * numSimTensors
    end


    # set main directory path
    mainDirPath = @sprintf("numFiles_%s/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", tensorUpdateFlag, latticeName, modelName, physicalSpin, setSym, Lx, Ly)

    # construct folderStringExpectValues
    folderStringExpectValues = mainDirPath * "/expectationValues_MF/bondDim_" * string(chiB)
    ~isdir(folderStringExpectValues) && mkpath(folderStringExpectValues)

    # construct fileStrings
    if occursin("PESS", tensorUpdateFlag)
        fileStringExpectValues = @sprintf("%s/expVals_hamVars_", folderStringExpectValues)
        for varIdx = eachindex(modelParameters)
            if varIdx < length(modelParameters)
                fileStringExpectValues *= @sprintf("%+0.2f_", modelParameters[varIdx])
            else
                fileStringExpectValues *= @sprintf("%+0.3f_", modelParameters[varIdx])
            end
        end
        fileStringExpectValues *= @sprintf("chiB_%d.jld", chiB)
    elseif occursin("PESO", tensorUpdateFlag)
        fileStringExpectValues = @sprintf("%s/expVals_hamVars_", folderStringExpectValues)
        for varIdx = eachindex(modelParameters)
            if varIdx < length(modelParameters)
                fileStringExpectValues *= @sprintf("%+0.2f_", modelParameters[varIdx])
            else
                fileStringExpectValues *= @sprintf("%+0.3f_", modelParameters[varIdx])
            end
        end
        fileStringExpectValues *= @sprintf("betaT_%0.2e_chiB_%d_dBeta_%0.2e.jld", β, chiB, δβ)
    end


    # construct folderString
    folderStringSimpleUpdate = mainDirPath * "/simpleUpdate/bondDim_" * string(chiB)

    # construct fileString
    if occursin("PESS", tensorUpdateFlag)
        fileStringSimpleUpdate = @sprintf("%s/simpleUpdate_hamVars_", folderStringSimpleUpdate)
        for varIdx = eachindex(modelParameters)
            if varIdx < length(modelParameters)
                fileStringSimpleUpdate *= @sprintf("%+0.2f_", modelParameters[varIdx])
            else
                fileStringSimpleUpdate *= @sprintf("%+0.3f_", modelParameters[varIdx])
            end
        end
        fileStringSimpleUpdate *= @sprintf("chiB_%d.jld", chiB)
    elseif occursin("PESO", tensorUpdateFlag)
        fileStringSimpleUpdate = @sprintf("%s/simpleUpdate_hamVars_", folderStringSimpleUpdate)
        for varIdx = eachindex(modelParameters)
            if varIdx < length(modelParameters)
                fileStringSimpleUpdate *= @sprintf("%+0.2f_", modelParameters[varIdx])
            else
                fileStringSimpleUpdate *= @sprintf("%+0.3f_", modelParameters[varIdx])
            end
        end
        fileStringSimpleUpdate *= @sprintf("betaT_%0.2e_chiB_%d_dBeta_%0.2e.jld", β, chiB, δβ)
    end

    if latticeName == "mapleLeafLattice"

        # construct fusing isomorphism
        fuseIsometry = TensorKit.isomorphism(fuse(vecSpacePhys, fuse(vecSpacePhys, fuse(vecSpacePhys, fuse(vecSpacePhys, fuse(vecSpacePhys, vecSpacePhys))))), vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys ⊗ vecSpacePhys)

        # constuct nearest-neighbour spin-spin interaction
        if setSym == "SU0"
            Sx, Sy, Sz, Sm, Sp, Id = getSpinOperators(physicalSpin)
            twoBodyGateSpinSpin = (Sx ⊗ Sx + Sy ⊗ Sy + Sz ⊗ Sz)
        end


        #----------------------------------------------------------------------
        # load simulation files
        #----------------------------------------------------------------------

        if tensorUpdateFlag == "iPESS"

            # load simTensorDicts, gamTensorDicts and lamTensorDicts
            simTensorDicts, gamTensorDicts, lamTensorDicts = load(fileStringSimpleUpdate, "simTensorDicts", "gamTensorDicts", "lamTensorDicts")
            println(fileStringSimpleUpdate)

            # convert TensorDicts to TensorMaps
            simTensors = convert.(TensorMap, simTensorDicts)
            gamTensors = convert.(TensorMap, gamTensorDicts)
            lamTensors = convert.(TensorMap, lamTensorDicts)

            # split dimer indices into six physical indices
            splitIsometry = TensorKit.isomorphism(vecSpacePhys ⊗ vecSpacePhys, fuse(vecSpacePhys, vecSpacePhys))
            splitGamTensors = Vector{TensorMap}(undef, length(gamTensorDicts))
            for idxG = eachindex(gamTensors)
                @tensor splitGamTensor[-1 -2 -3; -4] := splitIsometry[-1, -2, 1] * gamTensors[idxG][1, -3, -4]
                splitGamTensors[idxG] = splitGamTensor
            end

        elseif tensorUpdateFlag == "iPESO"

            # load simTensorDicts, gamTensorDicts and lamTensorDicts
            simTensorDicts, gamTensorDicts, lamTensorDicts = load(fileStringSimpleUpdate, "simTensorDicts", "gamTensorDicts", "lamTensorDicts")
            println(fileStringSimpleUpdate)

            # convert TensorDicts to TensorMaps
            simTensors = convert.(TensorMap, simTensorDicts)
            gamTensors = convert.(TensorMap, gamTensorDicts)
            lamTensors = convert.(TensorMap, lamTensorDicts)

            # split dimer indices into six physical indices
            splitIsometry = TensorKit.isomorphism(vecSpacePhys ⊗ vecSpacePhys, fuse(vecSpacePhys, vecSpacePhys))
            splitGamTensors = Vector{TensorMap}(undef, length(gamTensorDicts))
            for idxG = eachindex(gamTensors)
                @tensor splitGamTensor[-1 -2 -3; -4 -5 -6] := splitIsometry[-1, -2, 1] * gamTensors[idxG][1, -3, 2, -6] * splitIsometry'[2, -4, -5]
                splitGamTensors[idxG] = splitGamTensor
            end

        end


        # set number of lattice sites per unit cell
        numLatticeSitesPerUC = 6

        # initialize array to spin-spin correlation and bond energies
        listOfBondExpVals = zeros(Float64, 0, 4)

        # initialize array to store ground state magnetization
        listOfMagnetizations = zeros(Float64, 0, 4)

        # loop over all unit cells on the Square-Kagome lattice
        for idx = 1 : Lx, idy = 1 : Ly

            if tensorUpdateFlag == "iPESS"


                #----------------------------------------------------------------
                # simplex △
                #----------------------------------------------------------------

                # assignment for simplex △
                simNumber = getTensorNumber(idx + 0, idy + 0, unitCell, numSimTensors) + 1
                gamNumbers = [getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 1, getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 2, getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 3]'
                lamNumbers = [getTensorNumber(idx - 0, idy - 1, unitCell, numLamTensors) + 6, getTensorNumber(idx + 1, idy + 0, unitCell, numLamTensors) + 5, getTensorNumber(idx + 0, idy + 0, unitCell, numLamTensors) + 4]'

                # compute one-site reduced density matrix
                @tensor combinedSimplexU[-1 -2 -3 -4 -5 -6; -7 -8 -9] := lamTensors[lamNumbers[1]][-7, 1] * splitGamTensors[gamNumbers[1]][-1, -2, 1, 4] * lamTensors[lamNumbers[2]][-8, 2] * splitGamTensors[gamNumbers[2]][-3, -4, 2, 5] * simTensors[simNumber][4, 5, 6] * splitGamTensors[gamNumbers[3]][-5, -6, 6, 3] * lamTensors[lamNumbers[3]][3, -9]
                @tensor oneSiteRDM[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := combinedSimplexU[-1, -2, -3, -4, -5, -6, 1, 2, 3] * conj(combinedSimplexU[-7, -8, -9, -10, -11, -12, 1, 2, 3])
                oneSiteNorm = real(tr(oneSiteRDM))


                #----------------------------------------------------------------
                # compute magnetizations
                #----------------------------------------------------------------
            
                if setSym == "SU0"

                    # tensor 1 mod UC
                    gammaNum = 1 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[-1, 2, 3, 4, 5, 6, -2, 2, 3, 4, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 2 mod UC
                    gammaNum = 2 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, -1, 3, 4, 5, 6, 1, -2, 3, 4, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 3 mod UC
                    gammaNum = 3 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, -1, 4, 5, 6, 1, 2, -2, 4, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 4 mod UC
                    gammaNum = 4 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, 3, -1, 5, 6, 1, 2, 3, -2, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 5 mod UC
                    gammaNum = 5 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, 3, 4, -1, 6, 1, 2, 3, 4, -2, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 6 mod UC
                    gammaNum = 6 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, 3, 4, 5, -1, 1, 2, 3, 4, 5, -2]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # sort listOfMagnetizations according to lattice sites
                    listOfMagnetizations = sortslices(listOfMagnetizations, dims = 1)

                end


                #----------------------------------------------------------------
                # inter-unit-cell terms for simplex △
                #----------------------------------------------------------------

                # bond 1-2 mod UC
                gammaNums = [1 2] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[-1, -2, 3, 4, 5, 6, -3, -4, 3, 4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))

                # bond 1-5 mod UC
                gammaNums = [1 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[-1, 2, 3, 4, -2, 6, -3, 2, 3, 4, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

                # bond 2-3 mod UC
                gammaNums = [2 3] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, -2, 4, 5, 6, 1, -3, -4, 4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

                # bond 2-4 mod UC
                gammaNums = [2 4] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, 3, -2, 5, 6, 1, -3, 3, -4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

                # bond 2-5 mod UC
                gammaNums = [2 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, 3, 4, -2, 6, 1, -3, 3, 4, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

                # bond 3-4 mod UC
                gammaNums = [3 4] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, -1, -2, 5, 6, 1, 2, -3, -4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))

                # bond 4-5 mod UC
                gammaNums = [4 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, -1, -2, 6, 1, 2, 3, -3, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

                # bond 4-6 mod UC
                gammaNums = [4 6] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, -1, 5, -2, 1, 2, 3, -3, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

                # bond 5-6 mod UC
                gammaNums = [5 6] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, 4, -1, -2, 1, 2, 3, 4, -3, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))


                #----------------------------------------------------------------
                # intra-unit-cell terms for simplex ▽
                #----------------------------------------------------------------

                # select tensor numbers for position 1, 2, ..., 6
                tensorNumbers = [
                    getTensorNumber(idx + 1, idy + 1, unitCell, numLatticeSitesPerUC) + 1, 
                    getTensorNumber(idx + 1, idy + 1, unitCell, numLatticeSitesPerUC) + 2, 
                    getTensorNumber(idx + 0, idy + 0, unitCell, numLatticeSitesPerUC) + 3, 
                    getTensorNumber(idx + 0, idy + 0, unitCell, numLatticeSitesPerUC) + 4, 
                    getTensorNumber(idx + 1, idy + 0, unitCell, numLatticeSitesPerUC) + 5, 
                    getTensorNumber(idx + 1, idy + 0, unitCell, numLatticeSitesPerUC) + 6
                ]

                # assignment for simplex ▽
                simNumber = getTensorNumber(idx + 0, idy + 0, unitCell, numSimTensors) + 2
                gamNumbers = [getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 3, getTensorNumber(idx - 1, idy - 0, unitCell, numGamTensors) + 2, getTensorNumber(idx + 0, idy + 1, unitCell, numGamTensors) + 1]'
                lamNumbers = [getTensorNumber(idx + 0, idy + 0, unitCell, numLamTensors) + 3, getTensorNumber(idx - 1, idy - 0, unitCell, numLamTensors) + 2, getTensorNumber(idx + 0, idy + 1, unitCell, numLamTensors) + 1]'

                # compute three-site reduced density matrix
                @tensor combinedSimplexD[-1 -2 -3 -4 -5 -6; -7 -8 -9] := lamTensors[lamNumbers[1]][-7, 1] * splitGamTensors[gamNumbers[1]][-5, -6, 1, 4] * simTensors[simNumber][4, 5, 6] * splitGamTensors[gamNumbers[2]][-3, -4, 5, 2] * lamTensors[lamNumbers[2]][2, -8] * splitGamTensors[gamNumbers[3]][-1, -2, 6, 3] * lamTensors[lamNumbers[3]][3, -9]
                @tensor threeSiteRDM[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := combinedSimplexD[-1, -2, -3, -4, -5, -6, 1, 2, 3] * conj(combinedSimplexD[-7, -8, -9, -10, -11, -12, 1, 2, 3])
                threeSiteNorm = real(tr(threeSiteRDM))
                
                # bond 1-3 mod UC
                gammaNums = [tensorNumbers[1] tensorNumbers[3]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, -2, 4, 5, 6, -3, 2, -4, 4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))

                # bond 1-4 mod UC
                gammaNums = [tensorNumbers[1] tensorNumbers[4]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, 3, -2, 5, 6, -3, 2, 3, -4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

                # bond 1-6 mod UC
                gammaNums = [tensorNumbers[1] tensorNumbers[6]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, 3, 4, 5, -2, -3, 2, 3, 4, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))

                # bond 2-6 mod UC
                gammaNums = [tensorNumbers[2] tensorNumbers[6]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, -1, 3, 4, 5, -2, 1, -3, 3, 4, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

                # bond 3-5 mod UC
                gammaNums = [tensorNumbers[3] tensorNumbers[5]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, 2, -1, 4, -2, 6, 1, 2, -3, 4, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

                # bond 3-6 mod UC
                gammaNums = [tensorNumbers[3] tensorNumbers[6]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, 2, -1, 4, 5, -2, 1, 2, -3, 4, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))


            elseif tensorUpdateFlag == "iPESO"


                #----------------------------------------------------------------
                # simplex △
                #----------------------------------------------------------------

                # assignment for simplex △
                simNumber = getTensorNumber(idx + 0, idy + 0, unitCell, numSimTensors) + 1
                gamNumbers = [getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 1, getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 2, getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 3]'
                lamNumbers = [getTensorNumber(idx - 0, idy - 1, unitCell, numLamTensors) + 6, getTensorNumber(idx + 1, idy + 0, unitCell, numLamTensors) + 5, getTensorNumber(idx + 0, idy + 0, unitCell, numLamTensors) + 4]'

                # compute one-site reduced density matrix
                @tensor combinedSimplexU[-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14; -15] := lamTensors[lamNumbers[1]][-13, 1] * splitGamTensors[gamNumbers[1]][-1, -2, 1, -7, -8, 4] * lamTensors[lamNumbers[2]][-14, 2] * splitGamTensors[gamNumbers[2]][-3, -4, 2, -9, -10, 5] * simTensors[simNumber][4, 5, 6] * splitGamTensors[gamNumbers[3]][-5, -6, 6, -11, -12, 3] * lamTensors[lamNumbers[3]][3, -15]
                @tensor oneSiteRDM[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := combinedSimplexU[-1, -2, -3, -4, -5, -6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * conj(combinedSimplexU[-7, -8, -9, -10, -11, -12, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                oneSiteNorm = real(tr(oneSiteRDM))


                #----------------------------------------------------------------
                # compute magnetizations
                #----------------------------------------------------------------
            
                if setSym == "SU0"

                    # tensor 1 mod UC
                    gammaNum = 1 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[-1, 2, 3, 4, 5, 6, -2, 2, 3, 4, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 2 mod UC
                    gammaNum = 2 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, -1, 3, 4, 5, 6, 1, -2, 3, 4, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 3 mod UC
                    gammaNum = 3 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, -1, 4, 5, 6, 1, 2, -2, 4, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 4 mod UC
                    gammaNum = 4 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, 3, -1, 5, 6, 1, 2, 3, -2, 5, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 5 mod UC
                    gammaNum = 5 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, 3, 4, -1, 6, 1, 2, 3, 4, -2, 6]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # tensor 6 mod UC
                    gammaNum = 6 + getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                    expValsMag = zeros(Float64, 0)
                    for localOperator = [Sx, Sy, Sz]
                        @tensor oneSiteRDM_traced[-1; -2] := oneSiteRDM[1, 2, 3, 4, 5, -1, 1, 2, 3, 4, 5, -2]
                        oneSiteGate = tr(localOperator * oneSiteRDM_traced)
                        expValsMag = vcat(expValsMag, real(oneSiteGate / oneSiteNorm))
                    end
                    listOfMagnetizations = vcat(listOfMagnetizations, hcat(gammaNum, expValsMag'))

                    # sort listOfMagnetizations according to lattice sites
                    listOfMagnetizations = sortslices(listOfMagnetizations, dims = 1)

                end


                #----------------------------------------------------------------
                # inter-unit-cell terms for simplex △
                #----------------------------------------------------------------

                # bond 1-2 mod UC
                gammaNums = [1 2] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[-1, -2, 3, 4, 5, 6, -3, -4, 3, 4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))

                # bond 1-5 mod UC
                gammaNums = [1 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[-1, 2, 3, 4, -2, 6, -3, 2, 3, 4, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

                # bond 2-3 mod UC
                gammaNums = [2 3] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, -2, 4, 5, 6, 1, -3, -4, 4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

                # bond 2-4 mod UC
                gammaNums = [2 4] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, 3, -2, 5, 6, 1, -3, 3, -4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

                # bond 2-5 mod UC
                gammaNums = [2 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, -1, 3, 4, -2, 6, 1, -3, 3, 4, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

                # bond 3-4 mod UC
                gammaNums = [3 4] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, -1, -2, 5, 6, 1, 2, -3, -4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))

                # bond 4-5 mod UC
                gammaNums = [4 5] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, -1, -2, 6, 1, 2, 3, -3, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[4]))

                # bond 4-6 mod UC
                gammaNums = [4 6] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, -1, 5, -2, 1, 2, 3, -3, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[2]))

                # bond 5-6 mod UC
                gammaNums = [5 6] .+ getTensorNumber(idx, idy, unitCell, numLatticeSitesPerUC)
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := oneSiteRDM[1, 2, 3, 4, -1, -2, 1, 2, 3, 4, -3, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / oneSiteNorm, modelParameters[1]))


                #----------------------------------------------------------------
                # intra-unit-cell terms for simplex ▽
                #----------------------------------------------------------------

                # select tensor numbers for position 1, 2, ..., 6
                tensorNumbers = [
                    getTensorNumber(idx + 1, idy + 1, unitCell, numLatticeSitesPerUC) + 1, 
                    getTensorNumber(idx + 1, idy + 1, unitCell, numLatticeSitesPerUC) + 2, 
                    getTensorNumber(idx + 0, idy + 0, unitCell, numLatticeSitesPerUC) + 3, 
                    getTensorNumber(idx + 0, idy + 0, unitCell, numLatticeSitesPerUC) + 4, 
                    getTensorNumber(idx + 1, idy + 0, unitCell, numLatticeSitesPerUC) + 5, 
                    getTensorNumber(idx + 1, idy + 0, unitCell, numLatticeSitesPerUC) + 6
                ]

                # assignment for simplex ▽
                simNumber = getTensorNumber(idx + 0, idy + 0, unitCell, numSimTensors) + 2
                gamNumbers = [getTensorNumber(idx + 0, idy + 0, unitCell, numGamTensors) + 3, getTensorNumber(idx - 1, idy - 0, unitCell, numGamTensors) + 2, getTensorNumber(idx + 0, idy + 1, unitCell, numGamTensors) + 1]'
                lamNumbers = [getTensorNumber(idx + 0, idy + 0, unitCell, numLamTensors) + 3, getTensorNumber(idx - 1, idy - 0, unitCell, numLamTensors) + 2, getTensorNumber(idx + 0, idy + 1, unitCell, numLamTensors) + 1]'

                # compute three-site reduced density matrix
                @tensor combinedSimplexD[-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13; -14 -15] := lamTensors[lamNumbers[1]][-13, 1] * splitGamTensors[gamNumbers[1]][-5, -6, 1, -11, -12, 4] * simTensors[simNumber][4, 5, 6] * splitGamTensors[gamNumbers[2]][-3, -4, 5, -9, -10, 2] * lamTensors[lamNumbers[2]][2, -14] * splitGamTensors[gamNumbers[3]][-1, -2, 6, -7, -8, 3] * lamTensors[lamNumbers[3]][3, -15]
                @tensor threeSiteRDM[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := combinedSimplexD[-1, -2, -3, -4, -5, -6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * conj(combinedSimplexD[-7, -8, -9, -10, -11, -12, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                threeSiteNorm = real(tr(threeSiteRDM))
                
                # bond 1-3 mod UC
                gammaNums = [tensorNumbers[1] tensorNumbers[3]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, -2, 4, 5, 6, -3, 2, -4, 4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))

                # bond 1-4 mod UC
                gammaNums = [tensorNumbers[1] tensorNumbers[4]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, 3, -2, 5, 6, -3, 2, 3, -4, 5, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

                # bond 1-6 mod UC
                gammaNums = [tensorNumbers[1] tensorNumbers[6]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[-1, 2, 3, 4, 5, -2, -3, 2, 3, 4, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))

                # bond 2-6 mod UC
                gammaNums = [tensorNumbers[2] tensorNumbers[6]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, -1, 3, 4, 5, -2, 1, -3, 3, 4, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

                # bond 3-5 mod UC
                gammaNums = [tensorNumbers[3] tensorNumbers[5]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, 2, -1, 4, -2, 6, 1, 2, -3, 4, -4, 6]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[3]))

                # bond 3-6 mod UC
                gammaNums = [tensorNumbers[3] tensorNumbers[6]]
                @tensor twoSiteRDM_traced[-1 -2; -3 -4] := threeSiteRDM[1, 2, -1, 4, 5, -2, 1, 2, -3, 4, 5, -4]
                twoSiteGate = real(tr(twoBodyGateSpinSpin * twoSiteRDM_traced))
                listOfBondExpVals = vcat(listOfBondExpVals, hcat(gammaNums, twoSiteGate / threeSiteNorm, modelParameters[5]))

            end

        end

    end

    # sort listOfMagnetizations according to lattice sites
    listOfMagnetizations = sortslices(listOfMagnetizations, dims = 1)


    #----------------------------------------------------------------
    # compute bond energies
    #----------------------------------------------------------------

    bondEnergies = zeros(Float64, size(listOfBondExpVals, 1))
    disconCorrel = zeros(Float64, size(listOfBondExpVals, 1))
    for bondIdx = axes(listOfBondExpVals, 1)
        bondInfo = convert.(Int, listOfBondExpVals[bondIdx, 1 : 2])
        expValSzA = listOfMagnetizations[bondInfo[1], 4]
        expValSzB = listOfMagnetizations[bondInfo[2], 4]
        bondEnergies[bondIdx] = prod(listOfBondExpVals[bondIdx, 3 : 4]) - 1/5 * modelParameters[6] * (expValSzA + expValSzB)
        disconCorrel[bondIdx] = listOfBondExpVals[bondIdx, 3] - (expValSzA * expValSzB)
    end
    listOfBondExpVals = hcat(listOfBondExpVals, bondEnergies, disconCorrel)
    
    display(listOfBondExpVals)
    averageBondEnergy = 15/6 * sum(listOfBondExpVals[:, 5]) / size(listOfBondExpVals, 1)
    println(averageBondEnergy, "\n")
    
    display(listOfMagnetizations)
    magPerSite = sqrt.(listOfMagnetizations[:, 2] .^ 2 + listOfMagnetizations[:, 3] .^ 2 + listOfMagnetizations[:, 4] .^ 2)
    averageMagPerSite = 1 / length(magPerSite) * sum(magPerSite.^2)
    averageMagZ = 1 / size(listOfMagnetizations, 1) * sum(listOfMagnetizations[:, 4]) / physicalSpin
    println(averageMagPerSite)
    println(averageMagZ)
    println("\n")

    # store listOfBondExpVals, listOfBondVariances and listOfMagnetizations
    save(fileStringExpectValues, "listOfBondExpVals", listOfBondExpVals, "listOfMagnetizations", listOfMagnetizations)
    println(fileStringExpectValues)

end