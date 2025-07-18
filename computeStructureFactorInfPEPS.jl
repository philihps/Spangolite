# #!/usr/bin/env julia

# include functions
include("CTMRG.jl")
include("structureFactorCTMRG.jl")

# compute CTMRG structure factor
function computeStructureFactorInfPEPS(modelDict::Dict, tensorUpdateFlag::String; noiseFlag::Bool = true)

    # get model properties
    latticeName = modelDict["latticeName"]
    modelName = modelDict["modelName"]
    setSym = modelDict["setSym"]
    physicalSpin = modelDict["physicalSpin"]
    Lx = Int(modelDict["Lx"])
    Ly = Int(modelDict["Ly"])
    unitCell = modelDict["unitCell"]
    modelParameters = modelDict["modelParameters"]
    chiB = Int(modelDict["chiB"])
    chiE = Int(modelDict["chiE"])
    convTolE = modelDict["convTolE"]
    latticeVectors = modelDict["latticeVectors"]
    basisVectors = modelDict["basisVectors"]
    kx = modelDict["kx"]
    ky = modelDict["ky"]


    # get number of tensors in the unitCell
    uniqueTensors = unique(unitCell)
    numberOfUniqueTensors = length(uniqueTensors)

    # set main directory path
    if tensorUpdateFlag == "iPESS"
        mainDirPath = @sprintf("numFiles_%s/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", tensorUpdateFlag, latticeName, modelName, physicalSpin, setSym, Lx, Ly)
    elseif tensorUpdateFlag == "varPEPS"
        mainDirPath = @sprintf("numpyFiles/%s/S_%0.1f/%s/Lx_%d_Ly_%d", modelName, physicalSpin, setSym, Lx, Ly)
    end

    # set physVecSpace
    if modelName == "HeisenbergModel" || modelName == "Spangolite"
        if setSym == "SU0"
            physVecSpace = ComplexSpace(Int(2 * physicalSpin + 1))
        elseif setSym == "SU2"
            physVecSpace = SU2Space(physicalSpin => 1)
        end
    end


    #----------------------------------------------------------------------
    # load simulation files
    #----------------------------------------------------------------------

    # load simulation files
    if tensorUpdateFlag == "iPESS"

        # define number of simplex, gamma and lambda tensors per unit cell
        numSimTensors = 2
        numGamTensors = 3

        # construct folderString
        folderStringSimpleUpdate = mainDirPath * "/simpleUpdate/bondDim_" * string(chiB)

        # construct fileString
        fileStringSimpleUpdate = @sprintf("%s/simpleUpdate_hamVars_", folderStringSimpleUpdate)
        for varIdx = eachindex(modelParameters)
            if varIdx < length(modelParameters)
                fileStringSimpleUpdate *= @sprintf("%+0.2f_", modelParameters[varIdx])
            else
                fileStringSimpleUpdate *= @sprintf("%+0.3f_", modelParameters[varIdx])
            end
        end
        fileStringSimpleUpdate *= @sprintf("chiB_%d.jld", chiB)

        # load simTensorDicts, gamTensorDicts and lamTensorDicts
        simTensorDicts, gamTensorDicts, lamTensorDicts = load(fileStringSimpleUpdate, "simTensorDicts", "gamTensorDicts", "lamTensorDicts")
        println(fileStringSimpleUpdate)

        # convert TensorDicts to TensorMaps
        simTensors = convert.(TensorMap, simTensorDicts)
        gamTensors = convert.(TensorMap, gamTensorDicts)
        lamTensors = convert.(TensorMap, lamTensorDicts)

        # split dimer indices into two physical indices
        splitIsometry = TensorKit.isomorphism(physVecSpace ⊗ physVecSpace, fuse(physVecSpace, physVecSpace))
        splitGamTensors = Vector{TensorMap}(undef, length(gamTensors))
        for idxG = eachindex(gamTensors)
            @tensor splitGamTensor[-1 -2 -3; -4] := splitIsometry[-1, -2, 1] * gamTensors[idxG][1, -3, -4]
            splitGamTensors[idxG] = splitGamTensor
        end

        # coarse-grain each unit cell to an iPEPS tensor
        coarseGrainedTensorsRegular = Vector{TensorMap}(undef, numberOfUniqueTensors)
        coarseGrainedTensorsCombine = Vector{TensorMap}(undef, numberOfUniqueTensors)
        for idx = 1 : Lx, idy = 1 : Ly
            tensorNumber = getUnitCellNumber(idx, idy, unitCell)
            simNumbers = getTensorNumber(idx, idy, unitCell, numSimTensors) .+ collect(1 : 2)
            gamNumbers = getTensorNumber(idx, idy, unitCell, numGamTensors) .+ collect(1 : 3)
            cgTensorRegular, cgTensorCombine = coarseGrainMapleLeaf_iPESS(simTensors[simNumbers], splitGamTensors[gamNumbers])
            coarseGrainedTensorsRegular[tensorNumber] = cgTensorRegular
            coarseGrainedTensorsCombine[tensorNumber] = cgTensorCombine
        end
        
    end

    
    # -----------------------------------------------------------------------------------------------------
    # compute structure factor for momentumVec = [kx, ky]
    #-----------------------------------------------------------------------------------------------------

    # compute structure factor
    momentumVec = [kx, ky]
    display(reshape(momentumVec, 1, length(momentumVec)))
    println()
    structureFactor_C = zeros(ComplexF64, 3)
    structureFactor_D = zeros(ComplexF64, 3)
    for spinComponent = 1 : 3
        sf_C, sf_D = computeStructureFactor(unitCell, coarseGrainedTensorsCombine, momentumVec, [spinComponent, spinComponent], latticeVectors, basisVectors, chiE, convTolE)
        structureFactor_C[spinComponent] = sf_C
        structureFactor_D[spinComponent] = sf_D
    end

    display(structureFactor_C)
    display(structureFactor_D)
    display(sum(structureFactor_C))
    display(sum(structureFactor_D))
    
    
    #----------------------------------------------------------------------
    # store simulation files
    #----------------------------------------------------------------------

    # construct folderString
    folderStringStructureFactors = mainDirPath * "/structureFactors/bondDim_" * string(chiB)
    ~isdir(folderStringStructureFactors) && mkpath(folderStringStructureFactors)

    # construct folderString
    folderStringStructureFactor = @sprintf("%s/strucFac_hamVars_", folderStringStructureFactors)
    for varIdx = eachindex(modelParameters)
        if varIdx < length(modelParameters)
            folderStringStructureFactor *= @sprintf("%+0.2f_", modelParameters[varIdx])
        else
            folderStringStructureFactor *= @sprintf("%+0.3f_", modelParameters[varIdx])
        end
    end
    folderStringStructureFactor *= @sprintf("chiB_%d_chiE_%d", chiB, chiE)

    # construct fileString
    fileStringStructureFactor = @sprintf("%s/kx_%+0.3f_ky_%+0.3f.jld", folderStringStructureFactor, kx, ky)

    # store environmentTensorDicts
    save(fileStringStructureFactor, "structureFactor_C", structureFactor_C, "structureFactor_D", structureFactor_D)
    println(fileStringStructureFactor, "\n")

end