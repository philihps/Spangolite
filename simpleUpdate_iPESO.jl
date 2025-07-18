#!/usr/bin/env julia

# include functions
include("hosvd_iPESO.jl")
include("latticeSetupKagome.jl")
include("modelHamiltonians.jl")
include("utilities.jl")

# function to print simple update convergence information
function verbosePrintSimpleUpdate_iPESO(loopCounter::Int64, normSingularValue::Float64)
    @info("Simple Update", loopCounter, normSingularValue)
end

# simple update
function simpleUpdate_iPESO(modelDict::Dict, initMethod::Int64; verbosePrint = false)

    # get model properties
    latticeName = modelDict["latticeName"]
    modelName = modelDict["modelName"]
    setSym = modelDict["setSym"]
    physicalSpin = modelDict["physicalSpin"]
    Lx = Int(modelDict["Lx"])
    Ly = Int(modelDict["Ly"])
    unitCell = modelDict["unitCell"]
    modelParameters = modelDict["modelParameters"]
    inverseTemperatures = modelDict["invTemps"]
    chiB = Int(modelDict["chiB"])
    δβ = modelDict["dBeta"]

    # set main directory path
    mainDirPath = @sprintf("numFiles_iPESO/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", latticeName, modelName, physicalSpin, setSym, Lx, Ly)

    # set simpleUpdate directory path
    folderStringSimpleUpdate = mainDirPath * "/simpleUpdate/bondDim_" * string(chiB)
    folderStringTruncatError = mainDirPath * "/truncationError/bondDim_" * string(chiB)
    ~isdir(folderStringSimpleUpdate) && mkpath(folderStringSimpleUpdate)
    ~isdir(folderStringTruncatError) && mkpath(folderStringTruncatError)

    # set number of tensors in the iPESS unit cell
    numSimTensors = 2
    numGamTensors = 3
    numLamTensors = 2 * numGamTensors
    totalNumSimTensors = Lx * Ly * numSimTensors
    totalNumGamTensors = Lx * Ly * numGamTensors
    totalNumLamTensors = Lx * Ly * numLamTensors
    println("number of distinct links in the iPESS network = $totalNumLamTensors")

    # connstuct simple update configurations for the iPESS unit cell
    simLamAssignment, gamLamAssignment, listOfSimplexUpdates_3Site = generateSimplexUpdates(Lx, Ly, unitCell)

    # set physVecSpace and trivVecSpace
    if latticeName == "mapleLeafLattice"
        if setSym == "SU0"
            physVecSpace = fuse(ComplexSpace(Int(2 * physicalSpin + 1)), ComplexSpace(Int(2 * physicalSpin + 1)))
            trivVecSpace = ComplexSpace(1)
        end
    end

    # construct vector of chiB
    indBondDim = chiB * ones(Int64, totalNumLamTensors)

    # construct Suzuki-Trotter and Hamiltonian gates
    if latticeName == "mapleLeafLattice"
        if modelName == "Spangolite"
            if setSym == "SU0"
                invTempGates = getSuzukiTrotterGate_SP_iPESS_6Site_SU0(physicalSpin, modelParameters, δβ / 2)
                hamiltonianGate = getHamiltonianGates_SP_iPESS_6Site_SU0(physicalSpin, modelParameters)
            end
        end
    end
    
    
    # initialize iPESO unit cell
    if initMethod == 0
    
        # initialize tensors
        simTensors, gamTensors, lamTensors = initialize_iPESO(totalNumLamTensors, physVecSpace, trivVecSpace, simLamAssignment, gamLamAssignment)

        # set loop control parameters
        startBetaTIdx = 1
        evolvedAnnealingSteps = 0

    elseif initMethod == 1

        # find simulationFile with maximal β
        lastBetaTIdx = 1
        loadFileString = ""
        for betaTIdx = eachindex(inverseTemperatures)

            # get β
            β = inverseTemperatures[betaTIdx]

            # construct fileString
            fileStringSimpleUpdate = @sprintf("%s/simpleUpdate_hamVars_", folderStringSimpleUpdate)
            for varIdx = eachindex(modelParameters)
                if varIdx < length(modelParameters)
                    fileStringSimpleUpdate *= @sprintf("%+0.2f_", modelParameters[varIdx])
                else
                    fileStringSimpleUpdate *= @sprintf("%+0.3f_", modelParameters[varIdx])
                end
            end
            fileStringSimpleUpdate *= @sprintf("betaT_%0.2e_chiB_%d_dBeta_%0.2e.jld", β, chiB, δβ)

            # check if simulationFile exists
            if isfile(fileStringSimpleUpdate)
                lastBetaTIdx = betaTIdx
                loadFileString = fileStringSimpleUpdate
            else
                break
            end

        end

        # load simTensorDicts, gamTensorDicts and lamTensorDicts
        simTensorDicts, gamTensorDicts, lamTensorDicts = load(loadFileString, "simTensorDicts", "gamTensorDicts", "lamTensorDicts")
        simTensors = convert.(TensorMap, simTensorDicts)
        gamTensors = convert.(TensorMap, gamTensorDicts)
        lamTensors = convert.(TensorMap, lamTensorDicts)

        # set evolvedAnnealingSteps
        startBetaTIdx = lastBetaTIdx + 1
        evolvedAnnealingSteps = Integer(round(inverseTemperatures[lastBetaTIdx] / δβ))
        @show startBetaTIdx
        @show evolvedAnnealingSteps

    end

    # print simple update info
    @printf("\nRunning Simple Update...\n")

    # initialize list for singular values
    oldSingularValues = fill(0.0, maximum(indBondDim), totalNumLamTensors)

    # loop over inverse temperatures
    for betaTIdx = startBetaTIdx : length(inverseTemperatures)

        # get betaT
        β = inverseTemperatures[betaTIdx]


        # ------------------------------------------------------------------------------
        # run simple update
        # ------------------------------------------------------------------------------

        # determine number of numAnnealingSteps
        numAnnealingSteps = Integer(round(β / δβ)) - evolvedAnnealingSteps
        # println(numAnnealingSteps)

        # initialize matrix to store trunction errors
        truncErrors = zeros(Float64, 0, 3 * size(listOfSimplexUpdates_3Site, 1))

        # simple update
        normSingularValue = 1.0
        for loopCounter = 1 : numAnnealingSteps

            # evolve iPESO with infinitesimal temperature step δβ
            simTensors, gamTensors, lamTensors, truncError = trotterEvolvePESO(listOfSimplexUpdates_3Site, indBondDim, invTempGates, simTensors, gamTensors, lamTensors)
            
            # store truncation errors
            truncErrors = vcat(truncErrors, reshape(truncError, 1, length(truncError)))

            # get singular values
            newSingularValues = zeros(maximum(indBondDim), totalNumLamTensors)
            for lambdaIdx = eachindex(lamTensors)
                singularValues = getSingularValues(lamTensors[lambdaIdx])
                newSingularValues[1 : length(singularValues), lambdaIdx] = singularValues
            end

            # check convergence of singular values
            normSingularValue = norm(oldSingularValues - newSingularValues)

            # print convergence information
            verbosePrint && verbosePrintSimpleUpdate_iPESO(loopCounter, normSingularValue)

            # update singular values
            oldSingularValues = newSingularValues

        end

        # update numAnnealingSteps
        evolvedAnnealingSteps += numAnnealingSteps
       
        
        # ------------------------------------------------------------------------------
        # store simulation data
        # ------------------------------------------------------------------------------

        # convert TensorMaps to Dicts
        simTensorDicts = convert.(Dict, simTensors)
        gamTensorDicts = convert.(Dict, gamTensors)
        lamTensorDicts = convert.(Dict, lamTensors)
        
        # construct fileString
        fileStringSimpleUpdate = @sprintf("%s/simpleUpdate_hamVars_", folderStringSimpleUpdate)
        fileStringTruncatError = @sprintf("%s/truncatError_hamVars_", folderStringTruncatError)
        for varIdx = eachindex(modelParameters)
            if varIdx < length(modelParameters)
                fileStringSimpleUpdate *= @sprintf("%+0.2f_", modelParameters[varIdx])
                fileStringTruncatError *= @sprintf("%+0.2f_", modelParameters[varIdx])
            else
                fileStringSimpleUpdate *= @sprintf("%+0.3f_", modelParameters[varIdx])
                fileStringTruncatError *= @sprintf("%+0.3f_", modelParameters[varIdx])
            end
        end
        fileStringSimpleUpdate *= @sprintf("betaT_%0.2e_chiB_%d_dBeta_%0.2e.jld", β, chiB, δβ)
        fileStringTruncatError *= @sprintf("betaT_%0.2e_chiB_%d_dBeta_%0.2e.jld", β, chiB, δβ)

        # store gamTensorDicts, lamTensorDicts and singularValueTensor
        save(fileStringSimpleUpdate, "simTensorDicts", simTensorDicts, "gamTensorDicts", gamTensorDicts, "lamTensorDicts", lamTensorDicts)
        # println(fileStringSimpleUpdate)

        # store truncErrors
        save(fileStringTruncatError, "truncErrors", truncErrors)
        # println(fileStringTruncatError)

    end

end

function trotterEvolvePESO(listOfSimplexUpdates_3Site, indBondDim, invTempGates, simTensors, gamTensors, lamTensors)
    """ function to evolve iPESO with one infinitesimal temperature step with invTempGates """

    # initialize vector to store truncation error
    truncError = Float64[]

    # apply threeBodyGate to all simTensors △
    for updateIdx = axes(listOfSimplexUpdates_3Site, 1)

        # get update configuration
        updateConfig = listOfSimplexUpdates_3Site[updateIdx, :]
        simplexType = updateConfig[1]
        simNum = updateConfig[2]
        gamNums = updateConfig[3 : 5]
        lamNumsExt = updateConfig[6 : 8]
        lamNumsInt = updateConfig[9 : 11]

        # select gate for simplexType
        threeBodyGateKet = invTempGates[mod(simplexType + 1, 2) + 1]

        # use different update for △ and ▽ simplex
        if simplexType == 1

            #-----------------------------------------------------------------------
            # update triangle △
            #-----------------------------------------------------------------------

            # apply gate to simplex configuration
            @tensor theta[-1 -2 -3 -4 -5 -6 -7 -8 -9] := lamTensors[lamNumsExt[1]][-2, 1] * gamTensors[gamNums[1]][7, 1, -1, 4] * lamTensors[lamNumsExt[2]][-5, 2] * gamTensors[gamNums[2]][8, 2, -6, 5] * simTensors[simNum][4, 5, 6] * gamTensors[gamNums[3]][9, 6, -9, 3] * lamTensors[lamNumsExt[3]][3, -8] * threeBodyGateKet[-3, -4, -7, 7, 8, 9]
            theta /= norm(theta)

            # perform higher-order SVD
            nodeType = +1
            doTruncation = 1
            coreTensor, U, S, ϵ = hosvd_iPESO(theta, indBondDim[lamNumsInt], nodeType, doTruncation)
            truncError = vcat(truncError, ϵ)

            # restore unitary tensors
            U[1] = permute(U[1], (3, 2), (1, 4))
            U[2] = permute(U[2], (1, 2), (3, 4))
            U[3] = permute(U[3], (2, 1), (4, 3))

            # remove diagonal lambdaMatrices
            newU = Vector{TensorMap}(undef, length(U))
            @tensor newU[1][-1 -2; -3 -4] := pinv(lamTensors[lamNumsExt[1]])[-2, 2] * U[1][-1, 2, -3, -4]
            @tensor newU[2][-1 -2; -3 -4] := pinv(lamTensors[lamNumsExt[2]])[-2, 2] * U[2][-1, 2, -3, -4]
            @tensor newU[3][-1 -2; -3 -4] := U[3][-1, -2, -3, 4] * pinv(lamTensors[lamNumsExt[3]])[4, -4]

            # store updated tensors
            simTensors[simNum] = coreTensor
            setindex!(gamTensors, newU, gamNums)
            setindex!(lamTensors, S, lamNumsInt)

        end

    end

    # apply threeBodyGate to all simTensors ▽
    for updateIdx = axes(listOfSimplexUpdates_3Site, 1)

        # get update configuration
        updateConfig = listOfSimplexUpdates_3Site[updateIdx, :]
        simplexType = updateConfig[1]
        simNum = updateConfig[2]
        gamNums = updateConfig[3 : 5]
        lamNumsExt = updateConfig[6 : 8]
        lamNumsInt = updateConfig[9 : 11]

        # select gate for simplexType
        threeBodyGateKet = invTempGates[mod(simplexType + 1, 2) + 1]

        # use different update for △ and ▽ simplex
        if simplexType == 2

            #-----------------------------------------------------------------------
            # update triangle ▽
            #-----------------------------------------------------------------------

            # apply gate to simplex configuration
            @tensor theta[-1 -2 -3 -4 -5 -6 -7 -8 -9] := lamTensors[lamNumsExt[1]][-2, 1] * gamTensors[gamNums[1]][7, 1, -3, 4] * simTensors[simNum][4, 5, 6] * gamTensors[gamNums[2]][8, 5, -6, 2] * lamTensors[lamNumsExt[2]][2, -5] * gamTensors[gamNums[3]][9, 6, -7, 3] * lamTensors[lamNumsExt[3]][3, -8]  * threeBodyGateKet[-1, -4, -9, 7, 8, 9]
            theta /= norm(theta)

            # perform higher-order SVD
            nodeType = -1
            doTruncation = 1
            coreTensor, U, S, ϵ = hosvd_iPESO(theta, indBondDim[lamNumsInt], nodeType, doTruncation)
            truncError = vcat(truncError, ϵ)

            # restore unitary tensors
            U[1] = permute(U[1], (1, 2), (3, 4))
            U[2] = permute(U[2], (2, 1), (4, 3))
            U[3] = permute(U[3], (4, 1), (2, 3))

            # remove diagonal lambdaMatrices
            newU = Vector{TensorMap}(undef, length(U))
            @tensor newU[1][-1 -2; -3 -4] := pinv(lamTensors[lamNumsExt[1]])[-2, 2] * U[1][-1, 2, -3, -4]
            @tensor newU[2][-1 -2; -3 -4] := U[2][-1, -2, -3, 4] * pinv(lamTensors[lamNumsExt[2]])[4, -4]
            @tensor newU[3][-1 -2; -3 -4] := U[3][-1, -2, -3, 4] * pinv(lamTensors[lamNumsExt[3]])[4, -4]

            # store updated tensors
            simTensors[simNum] = coreTensor
            setindex!(gamTensors, newU, gamNums)
            setindex!(lamTensors, S, lamNumsInt)

        end

    end

    # function return
    return simTensors, gamTensors, lamTensors, truncError

end