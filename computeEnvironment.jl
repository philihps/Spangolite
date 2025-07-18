# #!/usr/bin/env julia

# include functions
include("CTMRG.jl")

# compute CTMRG environments
function computeEnvironment(modelDict::Dict, tensorUpdateFlag::String)

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
    convTolB = modelDict["convTolB"]
    convTolE = modelDict["convTolE"]

    # get number of tensors in the unitCell
    uniqueTensors = unique(unitCell)
    numberOfUniqueTensors = length(uniqueTensors)

    # set truncation threshold for CTMRG procedure
    truncBelowE = 1e-08

    # define number of simplex, gamma and lambda tensors per unit cell
    if tensorUpdateFlag == "iPESS" || tensorUpdateFlag == "iPESO"
        numSimTensors = 2
        numGamTensors = 3
        numLamTensors = 3 * numSimTensors
    end

    # set main directory path
    mainDirPath = @sprintf("numFiles_%s/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", tensorUpdateFlag, latticeName, modelName, physicalSpin, setSym, Lx, Ly)

    # set physVecSpace
    if modelName == "HeisenbergModel" || modelName == "Spangolite"
        if setSym == "SU0"
            physVecSpace = ComplexSpace(Int(2 * physicalSpin + 1))
        end
    end


    #----------------------------------------------------------------------
    # load simulation files
    #----------------------------------------------------------------------

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

    if tensorUpdateFlag == "iPESS"

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
    # run CTMRG
    #-----------------------------------------------------------------------------------------------------

    # run corner transfer matrix renormalization group scheme
    @printf("\nRunning CTMRG...\n")
    environmentTensors, numE = runCTMRG(unitCell, coarseGrainedTensorsCombine, chiE)

    # convert coarseGrainedTensorsRegular and coarseGrainedTensorsCombine to TensorDicts
    coarseGrainedTensorRegularDicts = convert.(Dict, coarseGrainedTensorsRegular)
    coarseGrainedTensorCombineDicts = convert.(Dict, coarseGrainedTensorsCombine)

    # convert environmentTensors to TensorDicts
    environmentTensorDicts = Vector{Vector{Dict}}(undef, numberOfUniqueTensors)
    for (idx, envTensor) = enumerate(environmentTensors)
        environmentTensorDicts[idx] = convert.(Dict, envTensor)
    end
    
    
    #----------------------------------------------------------------------
    # store simulation files
    #----------------------------------------------------------------------

    # construct folderString
    folderStringEnvironments = mainDirPath * "/environmentTensors/bondDim_" * string(chiB)
    ~isdir(folderStringEnvironments) && mkpath(folderStringEnvironments)

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

    # store environmentTensorDicts
    save(fileStringEnvironments, "coarseGrainedTensorRegularDicts", coarseGrainedTensorRegularDicts, "coarseGrainedTensorCombineDicts", coarseGrainedTensorCombineDicts, "environmentTensorDicts", environmentTensorDicts)
    println(fileStringEnvironments)

end