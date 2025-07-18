#!/usr/bin/env julia

# include functions
include("utilities.jl")

function initializeEnvironments(pepsTensors::Vector{<:TensorMap})

    # initialize CTMRG tensors
    environmentTensors = Vector{Vector{TensorMap}}(undef, length(pepsTensors))
    for tensorIdx = eachindex(pepsTensors)
        
        # get pepsTensor
        pepsTensor = pepsTensors[tensorIdx]
        elementType = eltype(pepsTensor)
        trivVecSpace = ComplexSpace(1)

        # initialize environment tensors
        nC1 = TensorMap(ones, trivVecSpace, trivVecSpace)
        nT1 = permute(TensorKit.id(trivVecSpace ⊗ space(pepsTensor, 4)'), (1, 2, 4), (3, ))

        nC2 = TensorMap(ones, trivVecSpace ⊗ trivVecSpace, one(trivVecSpace))
        nT2 = permute(TensorKit.id(space(pepsTensor, 5)' ⊗ trivVecSpace), (1, 3, 2), (4, ))

        nC3 = TensorMap(ones, trivVecSpace, trivVecSpace)
        nT3 = permute(TensorKit.id(trivVecSpace ⊗ space(pepsTensor, 2)'), (1, ), (2, 4, 3))
        
        nC4 = TensorMap(ones, one(trivVecSpace), trivVecSpace ⊗ trivVecSpace)
        nT4 = permute(TensorKit.id(trivVecSpace ⊗ space(pepsTensor, 1)'), (1, ), (3, 2, 4))
        
        # store environment tensors
        environmentTensors[tensorIdx] = [nC1, nT1, nC2, nT2, nC3, nT3, nC4, nT4]

    end
    return environmentTensors

end

function postProcessEnvTensor(envTensor)
    return envTensor /= norm(envTensor)
end

function getRhoTL(pepsTensor, envTensors)
    @tensor rhoUL[-1 -2 -3; -4 -5 -6] := envTensors[1][1, 2] * envTensors[2][2, 4, 6, -4] * envTensors[8][-1, 1, 3, 5] * pepsTensor[3, -2, 7, 4, -5] * conj(pepsTensor[5, -3, 7, 6, -6])
    return rhoUL
end

function getRhoBL(pepsTensor, envTensors)
    @tensor rhoBL[(); -1 -2 -3 -4 -5 -6] := envTensors[8][2, -1, 4, 6] * pepsTensor[4, 3, 7, -2, -4] * conj(pepsTensor[6, 5, 7, -3, -5]) * envTensors[7][2, 1] * envTensors[6][1, 3, 5, -6]
    return rhoBL
end

function getRhoTR(pepsTensor, envTensors)
    @tensor rhoTR[-1 -2 -3 -4 -5 -6; ()] := envTensors[2][-1, 3, 5, 1] * envTensors[3][1, 2] * pepsTensor[-2, -4, 7, 3, 4] * conj(pepsTensor[-3, -5, 7, 5, 6]) * envTensors[4][4, 6, -6, 2]
    return rhoTR
end

function getRhoBR(pepsTensor, envTensors)
    @tensor rhoBR[-1 -2 -3; -4 -5 -6] := pepsTensor[-1, 4, 7, -4, 3] * conj(pepsTensor[-2, 6, 7, -5, 5]) * envTensors[4][3, 5, 1, -6] * envTensors[6][-3, 4, 6, 2] * envTensors[5][2, 1]
    return rhoBR
end

function calculateQuarters(bulkTensors, environmentTensors)
    rhoTL = getRhoTL(bulkTensors[1, 1], environmentTensors[1, 1])
    rhoBL = getRhoBL(bulkTensors[2, 1], environmentTensors[2, 1])
    rhoTR = getRhoTR(bulkTensors[1, 2], environmentTensors[1, 2])
    rhoBR = getRhoBR(bulkTensors[2, 2], environmentTensors[2, 2])
    return rhoTL, rhoBL, rhoTR, rhoBR
end

function calculateProjectorsL(bulkTensors, environmentTensors, chiE::Int64, truncBelowE; projType::String = "F")
    
    # compute CTMRG quarters
    rhoTL, rhoBL, rhoTR, rhoBR = calculateQuarters(bulkTensors, environmentTensors)

    # compute rhoB and rhoT
    if projType == "F"
        rhoB = permute(rhoBR, (4, 5, 6), (1, 2, 3)) * permute(rhoBL, (4, 5, 6), (1, 2, 3))
        rhoT = permute(rhoTL, (1, 2, 3), (4, 5, 6)) * permute(rhoTR, (1, 2, 3), (4, 5, 6))
        rhoB /= norm(rhoB)
        rhoT /= norm(rhoT)
    end

    # compute biorthogonalization of rhoB * rhoT tensors and truncate UL, SL and VL
    BOL = rhoB * rhoT
    UL, SL, VL = tsvd(BOL, (1, 2, 3), (4, 5, 6), trunc = truncdim(chiE) & truncbelow(truncBelowE))
    SL /= norm(SL)
    sqrtSL = real(inv(sqrt(SL)))

    # build projectors for left truncation
    PLT = rhoT * VL' * sqrtSL
    PLB = sqrtSL * UL' * rhoB
    return [PLB, PLT]

end

function calculateProjectorsR(bulkTensors, environmentTensors, chiE::Int64, truncBelowE; projType::String = "F")
    
    # compute CTMRG quarters
    rhoTL, rhoBL, rhoTR, rhoBR = calculateQuarters(bulkTensors, environmentTensors)

    # compute rhoL and rhoR
    if projType == "F"
        rhoB = permute(rhoBL, (1, 2, 3), (4, 5, 6)) * permute(rhoBR, (1, 2, 3), (4, 5, 6))
        rhoT = permute(rhoTR, (4, 5, 6), (1, 2, 3)) * permute(rhoTL, (4, 5, 6), (1, 2, 3))
        rhoB /= norm(rhoB)
        rhoT /= norm(rhoT)
    end

    # compute biorthogonalization of rhoB * rhoT tensors and truncate UR, SR and VR
    BOR = rhoB * rhoT
    UR, SR, VR = tsvd(BOR, (1, 2, 3), (4, 5, 6), trunc = truncdim(chiE) & truncbelow(truncBelowE))
    SR /= norm(SR)
    sqrtSR = real(inv(sqrt(SR)))

    # build projectors for left truncation
    PTB = sqrtSR * UR' * rhoB
    PRT = rhoT * VR' * sqrtSR
    return [PTB, PRT]

end

function calculateProjectorsB(bulkTensors, environmentTensors, chiE::Int64, truncBelowE; projType::String = "F")
    
    # compute CTMRG quarters
    rhoTL, rhoBL, rhoTR, rhoBR = calculateQuarters(bulkTensors, environmentTensors)

    # compute rhoL and rhoR
    if projType == "F"
        rhoL = permute(rhoTL, (4, 5, 6), (1, 2, 3)) * permute(rhoBL, (1, 2, 3), (4, 5, 6))
        rhoR = permute(rhoBR, (1, 2, 3), (4, 5, 6)) * permute(rhoTR, (4, 5, 6), (1, 2, 3))
        rhoL /= norm(rhoL)
        rhoR /= norm(rhoR)
    end

    # compute biorthogonalization of rhoL * rhoR tensors and truncate UB, SB and VB
    BOB = rhoL * rhoR
    UB, SB, VB = tsvd(BOB, (1, 2, 3), (4, 5, 6), trunc = truncdim(chiE) & truncbelow(truncBelowE))
    SB /= norm(SB)
    sqrtSB = real(inv(sqrt(SB)))

    # build projectors for left truncation
    PBL = sqrtSB * UB' * rhoL
    PBR = rhoR * VB' * sqrtSB
    return [PBL, PBR]

end

function calculateProjectorsT(bulkTensors, environmentTensors, chiE::Int64, truncBelowE; projType::String = "F")
    
    # compute CTMRG quarters
    rhoTL, rhoBL, rhoTR, rhoBR = calculateQuarters(bulkTensors, environmentTensors)

    # compute rhoL and rhoR
    if projType == "F"
        rhoL = permute(rhoBL, (4, 5, 6), (1, 2, 3)) * permute(rhoTL, (1, 2, 3), (4, 5, 6))
        rhoR = permute(rhoTR, (1, 2, 3), (4, 5, 6)) * permute(rhoBR, (4, 5, 6), (1, 2, 3))
        rhoL /= norm(rhoL)
        rhoR /= norm(rhoR)
    end

    # compute biorthogonalization of rhoL * rhoR tensors and truncate UT, ST and VT
    BOT = rhoL * rhoR
    UT, ST, VT = tsvd(BOT, (1, 2, 3), (4, 5, 6), trunc = truncdim(chiE) & truncbelow(truncBelowE))
    ST /= norm(ST)
    sqrtST = real(inv(sqrt(ST)))

    # build projectors for left truncation
    PTL = sqrtST * UT' * rhoL
    PTR = rhoR * VT' * sqrtST
    return [PTL, PTR]

end

function iterateRows(unitCell; reverseIterator::Bool = false, onlyUnique::Bool = false)
    
    # get unitCell size
    Lx, Ly = size(unitCell)

    # construct initial iterator over the first direction of unitCell
    initialIteratorX = collect(1 : +1 : Lx)
    if reverseIterator
        reverse!(initialIteratorX)
    end

    # construct final iterator over the first direction of unitCell
    iteratorX = Int64[]
    storeUnique = Int64[]
    for iterX in initialIteratorX
        for iterY = 1 : Ly
            if onlyUnique
                indexUL = unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]
                if ~any(storeUnique .== indexUL)
                    if ~any(iteratorX .== iterX)
                        iteratorX = vcat(iteratorX, iterX)
                    end
                    storeUnique = vcat(storeUnique, indexUL)
                end
            else
                if ~any(iteratorX .== iterX)
                    iteratorX = vcat(iteratorX, iterX)
                end
            end
        end
    end
    return iteratorX

end

function iterateCols(unitCell; reverseIterator::Bool = false, onlyUnique::Bool = false)
    
    # get unitCell size
    Lx, Ly = size(unitCell)

    # construct initial iterator over the second direction of unitCell
    initialIteratorY = collect(1 : +1 : Ly)
    if reverseIterator
        reverse!(initialIteratorY)
    end

    # construct final iterator over the second direction of unitCell
    iteratorY = Int64[]
    storeUnique = Int64[]
    for iterY in initialIteratorY
        for iterX = 1 : Lx
            if onlyUnique
                indexUL = unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]
                if ~any(storeUnique .== indexUL)
                    if ~any(iteratorY .== iterY)
                        iteratorY = vcat(iteratorY, iterY)
                    end
                    storeUnique = vcat(storeUnique, indexUL)
                end
            else
                if ~any(iteratorY .== iterY)
                    iteratorY = vcat(iteratorY, iterY)
                end
            end
        end
    end
    return iteratorY

end

function absorptionL(unitCell, pepsTensors, environmentTensors, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the left environment """

    Lx, Ly = size(unitCell)

    # loop over all columns
    for iterY = iterateCols(unitCell, reverseIterator = false, onlyUnique = true)

        # compute all projectors for the absorption of one column
        projectorsL = Vector{Vector{TensorMap}}(undef, Lx)
        for iterX = 1 : Lx
            unitCellView = [unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 1, Ly)] ; unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 1, Ly)]]
            projectorsL[iterX] = calculateProjectorsL(pepsTensors[unitCellView], environmentTensors[unitCellView], chiE, truncBelowE)
        end

        # perform absorption of tensors
        newTensorsC4 = Vector{TensorMap}(undef, Lx)
        newTensorsT4 = Vector{TensorMap}(undef, Lx)
        newTensorsC1 = Vector{TensorMap}(undef, Lx)
        for iterX = 1 : Lx

            # select pepsTensor and envTensors
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors = environmentTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            # newC4
            truncationProjectorT = projectorsL[pIndex(iterX - 0, Lx)][2]
            @tensor newC4[(); -1 -2] := envTensors[7][2, 1] * envTensors[6][1, 3, 4, -2] * truncationProjectorT[2, 3, 4, -1]
            newTensorsC4[iterX] = postProcessEnvTensor(newC4)

            # newT4
            truncationProjectorB = projectorsL[pIndex(iterX - 0, Lx)][1]
            truncationProjectorT = projectorsL[pIndex(iterX - 1, Lx)][2]
            @tensor newT4[-1; -2 -3 -4] := truncationProjectorB[-1, 7, 8, 9] * envTensors[8][7, 1, 2, 4] * pepsTensor[2, 8, 6, 3, -3] * conj(pepsTensor[4, 9, 6, 5, -4]) * truncationProjectorT[1, 3, 5, -2]
            newTensorsT4[iterX] = postProcessEnvTensor(newT4)

            # newC1
            truncationProjectorB = projectorsL[pIndex(iterX - 1, Lx)][1]
            @tensor newC1[-1; -2] := envTensors[1][2, 1] * envTensors[2][1, 3, 4, -2] * truncationProjectorB[-1, 2, 3, 4]
            newTensorsC1[iterX] = postProcessEnvTensor(newC1)

        end

        # replace previous environment tensors with new ones
        for iterX = 1 : Lx
            environmentTensors[unitCell[iterX, pIndex(iterY + 1, Ly)]][1] = newTensorsC1[iterX]
            environmentTensors[unitCell[iterX, pIndex(iterY + 1, Ly)]][8] = newTensorsT4[iterX]
            environmentTensors[unitCell[iterX, pIndex(iterY + 1, Ly)]][7] = newTensorsC4[iterX]
        end

    end

    # function return
    return environmentTensors

end

function absorptionR(unitCell, pepsTensors, environmentTensors, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the right environment """

    Lx, Ly = size(unitCell)

    # loop over all columns
    for iterY = iterateCols(unitCell, reverseIterator = true, onlyUnique = true)

        # compute all projectors for the absorption of one column
        projectorsR = Vector{Vector{TensorMap}}(undef, Lx)
        for iterX = 1 : Lx
            unitCellView = [unitCell[pIndex(iterX + 0, Lx), pIndex(iterY - 1, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] ; unitCell[pIndex(iterX + 1, Lx), pIndex(iterY - 1, Ly)] unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 0, Ly)]]
            projectorsR[iterX] = calculateProjectorsR(pepsTensors[unitCellView], environmentTensors[unitCellView], chiE, truncBelowE)
        end

        # perform absorption of tensors
        newTensorsC2 = Vector{TensorMap}(undef, Lx)
        newTensorsT2 = Vector{TensorMap}(undef, Lx)
        newTensorsC3 = Vector{TensorMap}(undef, Lx)
        for iterX = 1 : Lx

            # select pepsTensor and envTensors
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors = environmentTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            # newC2
            truncationProjectorB = projectorsR[pIndex(iterX - 1, Lx)][1]
            @tensor newC2[(); -1 -2] := envTensors[2][-1, 2, 3, 1] * envTensors[3][1, 4] * truncationProjectorB[-2, 2, 3, 4]
            newTensorsC2[iterX] = postProcessEnvTensor(newC2)

            # newT2
            truncationProjectorT = projectorsR[pIndex(iterX - 1, Lx)][2]
            truncationProjectorB = projectorsR[pIndex(iterX - 0, Lx)][1]
            @tensor newT2[-1 -2 -3; -4] := truncationProjectorB[-3, 7, 8, 9] * pepsTensor[-1, 7, 6, 2, 3] * conj(pepsTensor[-2, 8, 6, 4, 5]) * envTensors[4][3, 5, 9, 1] * truncationProjectorT[2, 4, 1, -4]
            newTensorsT2[iterX] = postProcessEnvTensor(newT2)

            # newC3
            truncationProjectorT = projectorsR[pIndex(iterX - 0, Lx)][2]
            @tensor newC3[-1; -2] := envTensors[6][-1, 2, 3, 1] * envTensors[5][1, 4] * truncationProjectorT[2, 3, 4, -2]
            newTensorsC3[iterX] = postProcessEnvTensor(newC3)

        end

        # replace previous environment tensors with new ones
        for iterX = 1 : Lx
            environmentTensors[unitCell[iterX, pIndex(iterY - 1, Ly)]][3] = newTensorsC2[iterX]
            environmentTensors[unitCell[iterX, pIndex(iterY - 1, Ly)]][4] = newTensorsT2[iterX]
            environmentTensors[unitCell[iterX, pIndex(iterY - 1, Ly)]][5] = newTensorsC3[iterX]
        end

    end

    # function return
    return environmentTensors

end

function absorptionB(unitCell, pepsTensors, environmentTensors, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the bottom environment """

    Lx, Ly = size(unitCell)

    # loop over all rows
    for iterX = iterateRows(unitCell, reverseIterator = true, onlyUnique = true)

        # compute all projectors for the absorption of one row
        projectorsB = Vector{Vector{TensorMap}}(undef, Ly)
        for iterY = 1 : Ly
            unitCellView = [unitCell[pIndex(iterX - 1, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX - 1, Lx), pIndex(iterY + 1, Ly)] ; unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 1, Ly)]]
            projectorsB[iterY] = calculateProjectorsB(pepsTensors[unitCellView], environmentTensors[unitCellView], chiE, truncBelowE)
        end

        # perform absorption of tensors
        newTensorsC3 = Vector{TensorMap}(undef, Ly)
        newTensorsT3 = Vector{TensorMap}(undef, Ly)
        newTensorsC4 = Vector{TensorMap}(undef, Ly)
        for iterY = 1 : Ly

            # select pepsTensor and envTensors
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors = environmentTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            # newC3
            truncationProjectorL = projectorsB[pIndex(iterY - 0, Ly)][1]
            @tensor newC3[-1; -2] := truncationProjectorL[-1, 2, 3, 4] * envTensors[4][2, 3, 1, -2] * envTensors[5][4, 1]
            newTensorsC3[iterY] = postProcessEnvTensor(newC3)

            # newT3
            truncationProjectorR = projectorsB[pIndex(iterY - 0, Ly)][2]
            truncationProjectorL = projectorsB[pIndex(iterY - 1, Ly)][1]
            @tensor newT3[-1; -2 -3 -4] := truncationProjectorL[-1, 7, 8, 9] * envTensors[6][9, 2, 4, 1] * pepsTensor[7, 2, 6, -2, 3] * conj(pepsTensor[8, 4, 6, -3, 5]) * truncationProjectorR[3, 5, 1, -4]
            newTensorsT3[iterY] = postProcessEnvTensor(newT3)

            # newC4
            truncationProjectorR = projectorsB[pIndex(iterY - 1, Ly)][2]
            @tensor newC4[(); -1 -2] := envTensors[7][1, 4] * envTensors[8][1, -1, 2, 3] * truncationProjectorR[2, 3, 4, -2]
            newTensorsC4[iterY] = postProcessEnvTensor(newC4)

        end

        # replace previous environment tensors with new ones
        for iterY = 1 : Ly
            environmentTensors[unitCell[pIndex(iterX - 1, Lx), iterY]][5] = newTensorsC3[iterY]
            environmentTensors[unitCell[pIndex(iterX - 1, Lx), iterY]][6] = newTensorsT3[iterY]
            environmentTensors[unitCell[pIndex(iterX - 1, Lx), iterY]][7] = newTensorsC4[iterY]
        end

    end

    # function return
    return environmentTensors

end

function absorptionT(unitCell, pepsTensors, environmentTensors, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the top environment """

    Lx, Ly = size(unitCell)

    # loop over all rows
    for iterX = iterateRows(unitCell, reverseIterator = false, onlyUnique = true)

        # compute all projectors for the absorption of one row
        projectorsT = Vector{Vector{TensorMap}}(undef, Ly)
        for iterY = 1 : Ly
            unitCellView = [unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 1, Ly)] ; unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 1, Ly)]]
            projectorsT[iterY] = calculateProjectorsT(pepsTensors[unitCellView], environmentTensors[unitCellView], chiE, truncBelowE)
        end

        # perform absorption of tensors
        newTensorsC1 = Vector{TensorMap}(undef, Ly)
        newTensorsT1 = Vector{TensorMap}(undef, Ly)
        newTensorsC2 = Vector{TensorMap}(undef, Ly)
        for iterY = 1 : Ly

            # select pepsTensor and envTensors
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors = environmentTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            # newC1
            truncationProjectorR = projectorsT[pIndex(iterY - 1, Ly)][2]
            @tensor newC1[-1; -2] := envTensors[8][-1, 1, 3, 4] * envTensors[1][1, 2] * truncationProjectorR[2, 3, 4, -2]
            newTensorsC1[iterY] = postProcessEnvTensor(newC1)

            # newT1
            truncationProjectorL = projectorsT[pIndex(iterY - 1, Ly)][1]
            truncationProjectorR = projectorsT[pIndex(iterY - 0, Ly)][2]
            @tensor newT1[-1 -2 -3; -4] := truncationProjectorL[-1, 7, 8, 9] * envTensors[2][7, 2, 4, 1] * pepsTensor[8, -2, 6, 2, 3] * conj(pepsTensor[9, -3, 6, 4, 5]) * truncationProjectorR[1, 3, 5, -4]
            newTensorsT1[iterY] = postProcessEnvTensor(newT1)

            # newC2
            truncationProjectorL = projectorsT[pIndex(iterY - 0, Ly)][1]
            @tensor newC2[(); -1 -2] := envTensors[3][2, 1] * envTensors[4][3, 4, -2, 1] * truncationProjectorL[-1, 2, 3, 4]
            newTensorsC2[iterY] = postProcessEnvTensor(newC2)

        end

        # replace previous environment tensors with new ones
        for iterY = 1 : Ly
            environmentTensors[unitCell[pIndex(iterX + 1, Lx), iterY]][1] = newTensorsC1[iterY]
            environmentTensors[unitCell[pIndex(iterX + 1, Lx), iterY]][2] = newTensorsT1[iterY]
            environmentTensors[unitCell[pIndex(iterX + 1, Lx), iterY]][3] = newTensorsC2[iterY]
        end

    end

    # function return
    return environmentTensors

end

function calculateCornerSVDs(environmentTensors, chiE)
    cornerSingularValues = zeros(Float64, length(environmentTensors), 4, chiE)
    for envIdx = eachindex(environmentTensors)
        envTensors = environmentTensors[envIdx]
        for (cornerIdx, cornerNum) = enumerate([1, 3, 5, 7])
            F = tsvd(envTensors[cornerNum], (1, ), (2, ))
            singularVals = getSingularValues(F[2])
            cornerSingularValues[envIdx, cornerIdx, 1 : length(singularVals)] = singularVals
        end
    end
    return cornerSingularValues
end

function doAbsorptionStep(unitCell, pepsTensors, environmentTensors, chiE, truncbelowE)
    environmentTensors = absorptionL(unitCell, pepsTensors, environmentTensors, chiE, truncbelowE)
    environmentTensors = absorptionT(unitCell, pepsTensors, environmentTensors, chiE, truncbelowE)
    environmentTensors = absorptionR(unitCell, pepsTensors, environmentTensors, chiE, truncbelowE)
    environmentTensors = absorptionB(unitCell, pepsTensors, environmentTensors, chiE, truncbelowE)
    return environmentTensors
end

function runCTMRG(unitCell, pepsTensors, chiE; maxSteps::Int64 = 100, truncbelowE::Float64 = 1e-8, convTolE::Float64 = 1e-6, verbosePrint::Bool = true, printSummary::Bool = true)

    # initialize environmentTensors
    environmentTensors = initializeEnvironments(pepsTensors)

    # initialize cornerSingularValues to check convergence
    oldCornerSingularVals = zeros(Float64, length(pepsTensors), 4, chiE)

    # run CTMRG
    runCTMRG = true
    envLoopCounter = 1
    normSingularValues = 1
    while runCTMRG

        # do full CTMRG absorption step
        environmentTensors = doAbsorptionStep(unitCell, pepsTensors, environmentTensors, chiE, truncbelowE)

        # compute corner SVDs and check convergence
        newCornerSingularVals = calculateCornerSVDs(environmentTensors, chiE)
        normSingularValues = norm(newCornerSingularVals - oldCornerSingularVals)
        if (normSingularValues < convTolE) || (envLoopCounter == maxSteps)
            runCTMRG = false
        end
        oldCornerSingularVals = newCornerSingularVals

        # print CTMRG convergence info
        verbosePrint && @printf("CTMRG Step %03d - normSingularValues %0.8e\n", envLoopCounter, maximum(normSingularValues))

        # increase envLoopCounter
        envLoopCounter += 1

    end

    # return number of CTM steps necessary to achieve convergence
    numE = envLoopCounter - 1

    # print CTMRG convergence info
    printSummary && @printf("CTMRG converged with %d numSteps, normSingularValues %0.8e\n", numE, maximum(normSingularValues))

    # function return
    return environmentTensors, numE

end