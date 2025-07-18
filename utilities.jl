#!/usr/bin/env julia

# shift index periodically
pIndex(idx, arrayLength) = mod(idx - 1, arrayLength) + 1

# determine tensor number in the full unit cell
function getUnitCellNumber(idx::Int64, idy::Int64, unitCell::Matrix{Int64})
    unitCellLx, unitCellLy = size(unitCell)
    unitCellNumber = unitCell[pIndex(idx, unitCellLx), pIndex(idy, unitCellLy)]
    return unitCellNumber
end

# determine tensor number in the full unit cell
function getTensorNumber(idx::Int64, idy::Int64, unitCell::Matrix{Int64}, numIndTensorsInUnitCell::Int64)
    unitCellNumber = getUnitCellNumber(idx, idy, unitCell)
    tensorNumber = numIndTensorsInUnitCell * (unitCellNumber - 1)
    return tensorNumber
end

function getSpinOperators(physicalSpin::Float64)

    # initialize spin matrices
    d = Integer(2 * physicalSpin + 1)
    Sx = zeros(ComplexF64, d, d)
    Sy = zeros(ComplexF64, d, d)
    Sz = zeros(ComplexF64, d, d)

    # construct Sx, Sy and Sz
    for idx = 1 : d
        for idy = 1 : d

            entryXY = 0.5 * sqrt((physicalSpin + 1) * (idx + idy - 1) - idx * idy)

            if (idx + 1) == idy
                Sx[idx,idy] += entryXY
                Sy[idx,idy] -= 1im * entryXY
            end

            if idx == (idy + 1)
                Sx[idx,idy] += entryXY
                Sy[idx,idy] += 1im * entryXY
            end

            if idx == idy
                Sz[idx,idy] += physicalSpin + 1 - idx
            end

        end
    end

    # compute Sp, Sm and Id
    Sp = Sx + 1im * Sy
    Sm = Sx - 1im * Sy
    Id = one(Sz)

    # convert arrays to TensorMaps
    Sx = TensorMap(Sx, ℂ^d, ℂ^d)
    Sy = TensorMap(Sy, ℂ^d, ℂ^d)
    Sz = TensorMap(Sz, ℂ^d, ℂ^d)
    Sm = TensorMap(Sm, ℂ^d, ℂ^d)
    Sp = TensorMap(Sp, ℂ^d, ℂ^d)
    Id = TensorMap(Id, ℂ^d, ℂ^d)

    # return spin matrices
    return Sx, Sy, Sz, Sm, Sp, Id

end

# function to coarse-grain iPESS tensors to one iPEPS tensor
function coarseGrainMapleLeaf_iPESS(simTensors, gamTensors)

    # coarse-grain iPESS tensors to iPEPS tensor with three dimer indices
    @tensor coarseGrainedTensorRegular[-1 -2 -3 -4 -5 -6 -7 -8; -9 -10] := gamTensors[1][-3, -4, -1, 1] * gamTensors[2][-5, -6, -2, 2] * simTensors[1][1, 2, 3] * gamTensors[3][-7, -8, 3, 4] * simTensors[2][4, -9, -10]
    coarseGrainedTensorRegular /= norm(coarseGrainedTensorRegular)

    # fuse physical indices
    physSpace = space(coarseGrainedTensorRegular, 3)
    fuseIsometry = TensorKit.isomorphism(fuse(physSpace, fuse(physSpace, fuse(physSpace, fuse(physSpace, fuse(physSpace, physSpace))))), physSpace ⊗ physSpace ⊗ physSpace ⊗ physSpace ⊗ physSpace ⊗ physSpace)
    @tensor coarseGrainedTensorCombine[-1 -2 -3; -4 -5] := fuseIsometry[-3, 1, 2, 3, 4, 5, 6] * coarseGrainedTensorRegular[-1, -2, 1, 2, 3, 4, 5, 6, -4, -5]
    return coarseGrainedTensorRegular, coarseGrainedTensorCombine

end

function computeOneSiteRDM_iPEPS(pepsTensor, envTensors)
    @tensor oneSiteRDM[-1; -2] := envTensors[1][1, 2] * envTensors[2][2, 4, 15, 5] * envTensors[3][5, 6] *
        envTensors[8][11, 1, 3, 13] * pepsTensor[3, 12, -1, 4, 7] * conj(pepsTensor[13, 14, -2, 15, 16]) * envTensors[4][7, 16, 8, 6] *
        envTensors[7][11, 10] * envTensors[6][10, 12, 14, 9] * envTensors[5][9, 8]
    return oneSiteRDM
end

function computeOneSiteRDM_mapleLeaf(pepsTensor, envTensors)

    @tensor oneSiteRDM[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := envTensors[1][1, 2] * envTensors[2][2, 4, 15, 5] * envTensors[3][5, 6] *
        envTensors[8][11, 1, 3, 13] * pepsTensor[3, 12, -1, -2, -3, -4, -5, -6, 4, 7] * conj(pepsTensor[13, 14, -7, -8, -9, -10, -11, -12, 15, 16]) * envTensors[4][7, 16, 8, 6] *
        envTensors[7][11, 10] * envTensors[6][10, 12, 14, 9] * envTensors[5][9, 8]
    return oneSiteRDM

end

function computeThreeSiteRDM_mapleLeaf(bulkPEPSs, environmentTensors)
    
    envTensors = environmentTensors[1, 1]
    @tensor rhoUL[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10] := envTensors[1][1, 2] * envTensors[2][2, 10, 4, -8] * envTensors[8][-5, 1, 9, 3] * bulkPEPSs[1, 1][9, -6, 5, 6, -1, -2, 7, 8, 10, -9] * conj(bulkPEPSs[1, 1][3, -7, 5, 6, -3, -4, 7, 8, 4, -10])

    envTensors = environmentTensors[2, 1]
    @tensor rhoDL[-1 -2 -3 -4; -5 -6 -7 -8 -9 -10] := envTensors[8][2, -5, 9, 4] * bulkPEPSs[2, 1][9, 10, 5, 6, 7, 8, -1, -2, -6, -8] * conj(bulkPEPSs[2, 1][4, 3, 5, 6, 7, 8, -3, -4, -7, -9]) * envTensors[7][2, 1] * envTensors[6][1, 10, 3, -10]

    envTensors = environmentTensors[1, 2]
    @tensor rhoUR[-1 -2 -3 -4 -5 -6; ()] := envTensors[2][-1, 12, 4, 1] * envTensors[3][1, 2] * bulkPEPSs[1, 2][-2, -4, 5, 6, 7, 8, 9, 10, 12, 11] * conj(bulkPEPSs[1, 2][-3, -5, 5, 6, 7, 8, 9, 10, 4, 3]) * envTensors[4][11, 3, -6, 2]

    envTensors = environmentTensors[2, 2]
    @tensor rhoDR[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10] := bulkPEPSs[2, 2][-5, 9, -1, -2, 5, 6, 7, 8, -8, 10] * conj(bulkPEPSs[2, 2][-6, 3, -3, -4, 5, 6, 7, 8, -9, 4]) * envTensors[4][10, 4, 2, -10] * envTensors[6][-7, 9, 3, 1] * envTensors[5][1, 2]

    @tensor rhoL[-1 -2 -3 -4 -5 -6 -7 -8; -9 -10 -11 -12 -13 -14] := rhoDL[-3, -4, -7, -8, 1, 2, 3, -12, -13, -14] * rhoUL[-1, -2, -5, -6, 1, 2, 3, -9, -10, -11]
    @tensor rhoR[-1 -2 -3 -4 -5 -6 -7 -8 -9 -10; ()] := rhoUR[-5, -6, -7, 1, 2, 3] * rhoDR[-1, -2, -3, -4, -8, -9, -10, 1, 2, 3]
    @tensor threeSiteRDM[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := rhoL[-3, -4, -5, -6, -9, -10, -11, -12, 1, 2, 3, 4, 5, 6] * rhoR[-1, -2, -7, -8, 1, 2, 3, 4, 5, 6]
    return threeSiteRDM
    
end

function getSingularValues(inputTensor)
    singularValues = Vector{Float64}()
    blockSectors = blocksectors(inputTensor)
    for blockIdx = blockSectors
        append!(singularValues, real.(diag(block(inputTensor, blockIdx))))
    end
    return singularValues
end