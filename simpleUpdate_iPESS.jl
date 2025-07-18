#!/usr/bin/env julia

# include functions
include("hosvd_iPESS.jl")
include("latticeSetupKagome.jl")
include("modelHamiltonians.jl")
include("utilities.jl")

# function to print simple update convergence information
function verbosePrintSimpleUpdate_iPESS(loopCounter::Int64, controlTimeStep::Int64, normSingularValue::Float64, normSingularValuesConvergence::Float64)
    @info("Simple Update", loopCounter, controlTimeStep, normSingularValue, normSingularValuesConvergence)
end

# simple update
function simpleUpdate_iPESS(modelDict::Dict, initMethod::Int64; verbosePrint = false)

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
    convTolB = modelDict["convTolB"]

    # set main directory path
    mainDirPath = @sprintf("numFiles_iPESS/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", latticeName, modelName, physicalSpin, setSym, Lx, Ly)

    # set simpleUpdate directory path
    folderStringSimpleUpdate = mainDirPath * "/simpleUpdate/bondDim_" * string(chiB)
    ~isdir(folderStringSimpleUpdate) && mkpath(folderStringSimpleUpdate)

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
    if modelName == "HeisenbergModel" || modelName == "Spangolite"
        if setSym == "SU0"
            physVecSpace = fuse(ComplexSpace(Int(2 * physicalSpin + 1)), ComplexSpace(Int(2 * physicalSpin + 1)))
            trivVecSpace = ComplexSpace(1)
        end
    end

    # maximal number of steps
    maxNumSteps = Int(1e8)
    minIterationsPerGate = Int(1e2)
    maxIterationsPerGate = Int(1e6)
    
    # select flag to write output
    writeOutput = 1

    # construct vector of chiB
    indBondDim = chiB * ones(Int64, totalNumLamTensors)

    # initialize simulation refine parameter
    controlTimeStep = 1
    timeSteps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    gatesTimeSteps = Vector{Any}(undef, length(timeSteps))
    if modelName == "Spangolite"
        if setSym == "SU0"
            for gateIdx = eachindex(timeSteps)
                gatesTimeSteps[gateIdx] = getSuzukiTrotterGate_SP_iPESS_6Site_SU0(physicalSpin, modelParameters, timeSteps[gateIdx])
            end
        end
    end
    maxControlTimeStep = length(timeSteps)
        
    # initialize iPESS unit cell
    if initMethod == 0
        
        # initialize random tensors
        simTensors, gamTensors, lamTensors = initialize_iPESS(totalNumLamTensors, physVecSpace, trivVecSpace, simLamAssignment, gamLamAssignment)

    elseif initMethod == 1

        # change modelParameters to load existing data
        loadModelParameters = vcat(modelParameters[1 : (end - 1)], modelParameters[end] - 0.0)

        # set loadChiB
        loadChiB = chiB - 1

        # construct folderString
        folderStringLoadFiles = mainDirPath * "/simpleUpdate/bondDim_" * string(loadChiB)
        
        # construct fileString
        fileStringLoadFiles = @sprintf("%s/simpleUpdate_hamVars_", folderStringLoadFiles)
        for varIdx = eachindex(loadModelParameters)
            if varIdx < length(loadModelParameters)
                fileStringLoadFiles *= @sprintf("%+0.2f_", loadModelParameters[varIdx])
            else
                fileStringLoadFiles *= @sprintf("%+0.3f_", loadModelParameters[varIdx])
            end
        end
        fileStringLoadFiles *= @sprintf("chiB_%d.jld", loadChiB)

        # load simTensorDicts, gamTensorDicts and lamTensorDicts
        simTensorDicts, gamTensorDicts, lamTensorDicts = load(fileStringLoadFiles, "simTensorDicts", "gamTensorDicts", "lamTensorDicts")
        println(fileStringLoadFiles) 

        # convert TensorDicts to TensorMaps
        simTensors = convert.(TensorMap, simTensorDicts)
        gamTensors = convert.(TensorMap, gamTensorDicts)
        lamTensors = convert.(TensorMap, lamTensorDicts)

    end

    # initialize list for singular values
    oldSingularValues = fill(0.0, maximum(indBondDim), totalNumLamTensors)
    oldNormSingularValues = 0.0

    # initialize matrix to store trunction errors
    truncErrors = zeros(Float64, 0, 3 * size(listOfSimplexUpdates_3Site, 1))

    # print simple update info
    @printf("\nRunning Simple Update...\n")

    # truncate with fixed bond dimension or using truncation error
    useTruncErr = false

    # simple update parameter
    loopCounter = 1
    gateCounter = 1
    newNormSingularValues = 1.0
    normSingularValuesConvergence = 0.0
    runSimulation = 1
    while runSimulation == 1 && loopCounter <= maxNumSteps

        # select which two-body gate to use
        threeBodyGates = gatesTimeSteps[controlTimeStep]

        # initialize vector to store truncation error
        truncError = Float64[]

        # all different configurations for simplex updates
        for updateIdx = axes(listOfSimplexUpdates_3Site, 1)

            # get update configuration
            updateConfig = listOfSimplexUpdates_3Site[updateIdx, :]
            simplexType = updateConfig[1]
            simNum = updateConfig[2]
            gamNums = updateConfig[3 : 5]
            lamNumsExt = updateConfig[6 : 8]
            lamNumsInt = updateConfig[9 : 11]

            # select gate for simplexType
            threeBodyGate = threeBodyGates[mod(simplexType + 1, 2) + 1]

            # use different update for △ and ▽ simplex
            if simplexType == 1

                #-----------------------------------------------------------------------
                # update triangle △
                #-----------------------------------------------------------------------

                # apply gate to simplex configuration
                @tensor theta[-1 -2 -3 -4 -5 -6] := lamTensors[lamNumsExt[1]][-1, 1] * gamTensors[gamNums[1]][7, 1, 4] * lamTensors[lamNumsExt[2]][-4, 2] * gamTensors[gamNums[2]][8, 2, 5] *
                    gamTensors[gamNums[3]][9, 6, 3] * lamTensors[lamNumsExt[3]][3, -6] * simTensors[simNum][4, 5, 6] * threeBodyGate[-2, -3, -5, 7, 8, 9]
                theta /= norm(theta)

                # perform higher-order SVD
                nodeType = +1
                doTruncation = 1
                coreTensor, U, S, ϵ = hosvd_iPESS(theta, indBondDim[lamNumsInt], nodeType, doTruncation, useTruncErr = useTruncErr)
                truncError = vcat(truncError, ϵ)

                # restore unitary tensors
                U[1] = permute(U[1], (2, 1), (3, ))
                U[2] = permute(U[2], (1, 2), (3, ))
                U[3] = permute(U[3], (2, 1), (3, ))

                # remove diagonal lambdaMatrices
                newU = Vector{TensorMap}(undef, length(U))
                @tensor newU[1][-1 -2; -3] := pinv(lamTensors[lamNumsExt[1]])[-2, 2] * U[1][-1, 2, -3]
                @tensor newU[2][-1 -2; -3] := pinv(lamTensors[lamNumsExt[2]])[-2, 2] * U[2][-1, 2, -3]
                @tensor newU[3][-1 -2; -3] := U[3][-1, -2, 3] * pinv(lamTensors[lamNumsExt[3]])[3, -3]

                # store updated tensors
                setindex!(simTensors, coreTensor, simNum)
                setindex!(gamTensors, newU, gamNums)
                setindex!(lamTensors, S, lamNumsInt)

            elseif simplexType == 2

                #-----------------------------------------------------------------------
                # update triangle ▽
                #-----------------------------------------------------------------------

                # apply gate to simplex configuration
                @tensor theta[-1 -2 -3 -4 -5 -6] := lamTensors[lamNumsExt[1]][-2, 1] * gamTensors[gamNums[1]][7, 1, 4] * gamTensors[gamNums[2]][8, 5, 2] * lamTensors[lamNumsExt[2]][2, -4] *
                    gamTensors[gamNums[3]][9, 6, 3] * lamTensors[lamNumsExt[3]][3, -5] * simTensors[simNum][4, 5, 6] * threeBodyGate[-1, -3, -6, 7, 8, 9]
                theta /= norm(theta)

                # perform higher-order SVD
                nodeType = -1
                doTruncation = 1
                coreTensor, U, S, ϵ = hosvd_iPESS(theta, indBondDim[lamNumsInt], nodeType, doTruncation, useTruncErr = useTruncErr)
                truncError = vcat(truncError, ϵ)

                # restore unitary tensors
                U[1] = permute(U[1], (1, 2), (3, ))
                U[2] = permute(U[2], (2, 1), (3, ))
                U[3] = permute(U[3], (3, 1), (2, ))

                # remove diagonal lambdaMatrices
                newU = Vector{TensorMap}(undef, length(U))
                @tensor newU[1][-1 -2; -3] := pinv(lamTensors[lamNumsExt[1]])[-2, 1] * U[1][-1, 1, -3]
                @tensor newU[2][-1 -2; -3] := U[2][-1, -2, 1] * pinv(lamTensors[lamNumsExt[2]])[1, -3]
                @tensor newU[3][-1 -2; -3] := U[3][-1, -2, 1] * pinv(lamTensors[lamNumsExt[3]])[1, -3]

                # store updated tensors
                setindex!(simTensors, coreTensor, simNum)
                setindex!(gamTensors, newU, gamNums)
                setindex!(lamTensors, S, lamNumsInt)

            end

        end

        # store truncation errors
        truncErrors = vcat(truncErrors, reshape(truncError, 1, length(truncError)))


        #-----------------------------------------------------------------------
        # store singular values and determine convergence
        #-----------------------------------------------------------------------

        # store singular values
        newSingularValues = zeros(maximum(indBondDim), totalNumLamTensors)
        for lambdaIdx in eachindex(lamTensors)
            singularValues = getSingularValues(lamTensors[lambdaIdx])
            newSingularValues[1 : length(singularValues), lambdaIdx] = singularValues
        end

        # compute newNormSingularValues and normSingularValuesConvergence
        newNormSingularValues = norm(newSingularValues - oldSingularValues)
        normSingularValuesConvergence = norm(newNormSingularValues - oldNormSingularValues)

        # print convergence information
        verbosePrint && verbosePrintSimpleUpdate_iPESS(loopCounter, controlTimeStep, newNormSingularValues, normSingularValuesConvergence)

        # check convergence after sufficiently many simple update steps with each Trotter gate
        if gateCounter > minIterationsPerGate

            # check for convergence of singular values
            if (newNormSingularValues < convTolB) || gateCounter > maxIterationsPerGate
                @printf("ITE step %d converged in %d steps\n", controlTimeStep, gateCounter)
                if controlTimeStep < maxControlTimeStep
                    controlTimeStep += 1
                    gateCounter = 1
                else
                    runSimulation = 0
                end
            end

        end

        # update singular values
        oldSingularValues = newSingularValues

        # update normSingularValues
        oldNormSingularValues = newNormSingularValues

        # increase loopCounter
        loopCounter += 1
        gateCounter += 1

    end

    display(space.(lamTensors))

    # print simple update summary
    loopCounter -= 1
    @printf("\nSimple Update Completed with %d Steps\n", loopCounter)


    #----------------------------------------------------------------------
    # store simulation files
    #----------------------------------------------------------------------

    # convert TensorMaps to Dicts
    simTensorDicts = convert.(Dict, simTensors)
    gamTensorDicts = convert.(Dict, gamTensors)
    lamTensorDicts = convert.(Dict, lamTensors)

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

    # store simTensorDicts, gamTensorDicts and lamTensorDicts
    if writeOutput == 1
        save(fileStringSimpleUpdate, "simTensorDicts", simTensorDicts, "gamTensorDicts", gamTensorDicts, "lamTensorDicts", lamTensorDicts)
        println(fileStringSimpleUpdate, "\n")
    end

end