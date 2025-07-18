# function to generate all different configurations for simplex updates
function generateSimplexUpdates(Lx, Ly, unitCell)

    # number of tensors in the iPESS/iPESO unit cell
    numSimTensors = 2
    numGamTensors = 3
    numLamTensors = 2 * numGamTensors

    # get size of unitCell
    unitCellLx, unitCellLy = size(unitCell)

    # make unit cell periodic
    pIndex(idx, arrayLength) = mod(idx - 1, arrayLength) + 1

    function getSimNumber(unitCellIdx, unitCellIdy)
        unitCellNumber = unitCell[pIndex(unitCellIdx, unitCellLx), pIndex(unitCellIdy, unitCellLy)]
        simNumber = numSimTensors * (unitCellNumber - 1)
        return simNumber
    end

    function getGamNumber(unitCellIdx, unitCellIdy)
        unitCellNumber = unitCell[pIndex(unitCellIdx, unitCellLx), pIndex(unitCellIdy, unitCellLy)]
        gamNumber = numGamTensors * (unitCellNumber - 1)
        return gamNumber
    end

    function getLamNumber(unitCellIdx, unitCellIdy)
        unitCellNumber = unitCell[pIndex(unitCellIdx, unitCellLx), pIndex(unitCellIdy, unitCellLy)]
        LamNumber = numLamTensors * (unitCellNumber - 1)
        return LamNumber
    end

    # construct list of simplex tensors with corresponding lambda tensors
    simLamAssignment = zeros(Int64, 0, 4)
    for idx = 1 : Lx, idy = 1 : Ly

        # assignment for simplex △
        simplexNum = getSimNumber(idx + 0, idy + 0) + 1
        lambdaNums = getLamNumber(idx + 0, idy + 0) .+ [1, 2, 3]'
        simLamAssignment = vcat(simLamAssignment, hcat(simplexNum, lambdaNums))

        # assignment for simplex ▽
        simplexNum = getSimNumber(idx + 0, idy + 0) + 2
        lambdaNums = getLamNumber(idx + 0, idy + 0) .+ [4, 5, 6]'
        simLamAssignment = vcat(simLamAssignment, hcat(simplexNum, lambdaNums))

    end

    # construct list of gamma tensors with corresponding lambda tensors (1st lambda : outgoing virtual index, 2nd lambda : incoming virtual index)
    gamLamAssignment = zeros(Int64, 0, 3)
    for idx = 1 : Lx, idy = 1 : Ly

        # assignment for gamma tensor 1
        gammaNum = getGamNumber(idx + 0, idy + 0) + 1
        lambdaNums = [getLamNumber(idx - 0, idy - 1) + 6, getLamNumber(idx + 0, idy + 0) + 1]'
        gamLamAssignment = vcat(gamLamAssignment, hcat(gammaNum, lambdaNums))

        # assignment for gamma tensor 2
        gammaNum = getGamNumber(idx + 0, idy + 0) + 2
        lambdaNums = [getLamNumber(idx + 1, idy + 0) + 5, getLamNumber(idx - 0, idy - 0) + 2]'
        gamLamAssignment = vcat(gamLamAssignment, hcat(gammaNum, lambdaNums))

        # assignment for gamma tensor 3
        gammaNum = getGamNumber(idx + 0, idy + 0) + 3
        lambdaNums = [getLamNumber(idx + 0, idy + 0) + 3, getLamNumber(idx + 0, idy + 0) + 4]'
        gamLamAssignment = vcat(gamLamAssignment, hcat(gammaNum, lambdaNums))

    end

    # construct list of simplex updates for 3-site simple update
    listOfSimplexUpdates_3Site = zeros(Int64, 0, 5 + 3 + 3)
    for idx = 1 : Lx, idy = 1 : Ly

        # assignment for simplex △
        simplexNum = getSimNumber(idx + 0, idy + 0) + 1
        gamNumbers = [getGamNumber(idx + 0, idy + 0) + 1, getGamNumber(idx + 0, idy + 0) + 2, getGamNumber(idx + 0, idy + 0) + 3]'
        lamNumsOld = [getLamNumber(idx - 0, idy - 1) + 6, getLamNumber(idx + 1, idy + 0) + 5, getLamNumber(idx + 0, idy + 0) + 4]'
        lamNumsNew = [getLamNumber(idx + 0, idy + 0) + 1, getLamNumber(idx + 0, idy + 0) + 2, getLamNumber(idx + 0, idy + 0) + 3]'
        listOfSimplexUpdates_3Site = vcat(listOfSimplexUpdates_3Site, hcat(1, simplexNum, gamNumbers, lamNumsOld, lamNumsNew))

        # assignment for simplex ▽
        simplexNum = getSimNumber(idx + 0, idy + 0) + 2
        gamNumbers = [getGamNumber(idx + 0, idy + 0) + 3, getGamNumber(idx - 1, idy - 0) + 2, getGamNumber(idx + 0, idy + 1) + 1]'
        lamNumsOld = [getLamNumber(idx + 0, idy + 0) + 3, getLamNumber(idx - 1, idy - 0) + 2, getLamNumber(idx + 0, idy + 1) + 1]'
        lamNumsNew = [getLamNumber(idx + 0, idy + 0) + 4, getLamNumber(idx + 0, idy + 0) + 5, getLamNumber(idx + 0, idy + 0) + 6]'
        listOfSimplexUpdates_3Site = vcat(listOfSimplexUpdates_3Site, hcat(2, simplexNum, gamNumbers, lamNumsOld, lamNumsNew))

    end

    # function return
    return simLamAssignment, gamLamAssignment, listOfSimplexUpdates_3Site

end

function initialize_iPESS(numLinks, physVecSpace, virtVecSpace, simLamAssignment, gamLamAssignment)

    # get number of simplex and gamma tensors
    numSimTensors = size(simLamAssignment, 1)
    numGamTensors = size(gamLamAssignment, 1)

    # initialize simplexTensors
    simplexTensors = Vector{TensorMap}(undef, numSimTensors)
    for simIdx = 1 : numSimTensors
        simConfig = simLamAssignment[simIdx, :]
        simNumber = simConfig[1]
        if mod(simNumber, 2) == 0
            simplexTensors[simNumber] = TensorMap(randn, virtVecSpace, virtVecSpace ⊗ virtVecSpace)
        elseif mod(simNumber, 2) == 1
            simplexTensors[simNumber] = TensorMap(randn, virtVecSpace ⊗ virtVecSpace, virtVecSpace)
        end
    end

    # initialize gammaTensors
    gammaTensors = Vector{TensorMap}(undef, numGamTensors)
    for gamIdx = 1 : numGamTensors
        gamConfig = gamLamAssignment[gamIdx, :]
        gamNumber = gamConfig[1]
        gammaTensors[gamNumber] = TensorMap(randn, physVecSpace ⊗ virtVecSpace, virtVecSpace)
    end

    # initialize lambdaTensors
    lambdaTensors = Vector{TensorMap}(undef, numLinks)
    for lamIdx = 1 : numLinks
        lambdaTensors[lamIdx] = TensorKit.id(virtVecSpace)
    end

    # return iPESS unit cell
    return simplexTensors, gammaTensors, lambdaTensors

end

function initialize_iPESO(numLinks, physVecSpace, trivVecSpace, simLamAssignment, gamLamAssignment)

    # get number of simplex and gamma tensors
    numSimTensors = size(simLamAssignment, 1)
    numGamTensors = size(gamLamAssignment, 1)

    # initialize simplexTensors
    simplexTensors = Vector{TensorMap}(undef, numSimTensors)
    for simIdx = 1 : numSimTensors
        simConfig = simLamAssignment[simIdx, :]
        simNumber = simConfig[1]
        if mod(simNumber, 2) == 0
            simplexTensors[simNumber] = TensorMap(ones, trivVecSpace, trivVecSpace ⊗ trivVecSpace)
        elseif mod(simNumber, 2) == 1
            simplexTensors[simNumber] = TensorMap(ones, trivVecSpace ⊗ trivVecSpace, trivVecSpace)
        end
    end

    # initialize gammaTensors
    gammaTensors = Vector{TensorMap}(undef, numGamTensors)
    for gamIdx = 1 : numGamTensors
        gamConfig = gamLamAssignment[gamIdx, :]
        gamNumber = gamConfig[1]
        gammaTensors[gamNumber]  = TensorKit.id(physVecSpace ⊗ trivVecSpace)
    end

    # initialize lambdaTensors
    lambdaTensors = Vector{TensorMap}(undef, numLinks)
    for lamIdx = 1 : numLinks
        lambdaTensors[lamIdx] = TensorKit.id(trivVecSpace)
    end

    # return iPESO unit cell
    return simplexTensors, gammaTensors, lambdaTensors

end