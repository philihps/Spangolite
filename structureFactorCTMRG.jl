#!/usr/bin/env julia

# include functions
include("utilities.jl")

function getSpinOperatorFullBasis(basisSite::Int64, spinComponent::Int64; physicalSpin::Float64 = 0.5, basisSize::Int64 = 6)
    
    # get spin matrices for physical spin S
    Sx, Sy, Sz, Sm, Sp, Id = getSpinOperators(physicalSpin)
    spinComponents = [Sx, Sy, Sz]

    if basisSize == 1
        basisOperator = spinComponents[spinComponent]
    elseif basisSize == 6

        # construct spin operator in the basis of the lattice
        if basisSite == 1
            basisOperator = spinComponents[spinComponent] ⊗ Id ⊗ Id ⊗ Id ⊗ Id ⊗ Id
        elseif basisSite == 2
            basisOperator = Id ⊗ spinComponents[spinComponent] ⊗ Id ⊗ Id ⊗ Id ⊗ Id
        elseif basisSite == 3
            basisOperator = Id ⊗ Id ⊗ spinComponents[spinComponent] ⊗ Id ⊗ Id ⊗ Id
        elseif basisSite == 4
            basisOperator = Id ⊗ Id ⊗ Id ⊗ spinComponents[spinComponent] ⊗ Id ⊗ Id
        elseif basisSite == 5
            basisOperator = Id ⊗ Id ⊗ Id ⊗ Id ⊗ spinComponents[spinComponent] ⊗ Id
        elseif basisSite == 6
            basisOperator = Id ⊗ Id ⊗ Id ⊗ Id ⊗ Id ⊗ spinComponents[spinComponent]
        end

        # fuse individual indices into one index
        physSpace = ComplexSpace(Int(2 * physicalSpin + 1))
        fuseIsometry = TensorKit.isomorphism(fuse(physSpace, fuse(physSpace, fuse(physSpace, fuse(physSpace, fuse(physSpace, physSpace))))), physSpace ⊗ physSpace ⊗ physSpace ⊗ physSpace ⊗ physSpace ⊗ physSpace)
        @tensor basisOperator[-1; -2] := fuseIsometry[-1, 1, 2, 3, 4, 5, 6] * basisOperator[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * fuseIsometry'[7, 8, 9, 10, 11, 12, -2]

    end
    return basisOperator

end

function postProcessEnvTensor_sF(envTensor)
    tensorNorm = norm(envTensor)
    envTensor /= tensorNorm
    return envTensor, tensorNorm
end


function absorptionL_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors::Vector{Vector{Float64}}, globalPhase_L::ComplexF64, environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the left environment """

    Lx, Ly = size(unitCell)

    # loop over all columns
    for iterY = iterateCols(unitCell, reverseIterator = false, onlyUnique = true)

        # compute all projectors for the absorption of one column
        projectorsL = Vector{Vector{TensorMap}}(undef, Lx)
        for iterX = 1 : Lx
            unitCellView = [unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 1, Ly)] ; unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 1, Ly)]]
            projectorsL[iterX] = calculateProjectorsL(pepsTensors[unitCellView], environmentTensors_gS[unitCellView], chiE, truncBelowE)
        end

        #--------------------------------------------------------

        # perform absorption of tensors for environmentTensors_gS and environmentTensors_sF
        newTensorsC4_gS = Vector{TensorMap}(undef, Lx)
        newTensorsT4_gS = Vector{TensorMap}(undef, Lx)
        newTensorsC1_gS = Vector{TensorMap}(undef, Lx)
        normsTensorC4 = zeros(Float64, Lx)
        normsTensorT4 = zeros(Float64, Lx)
        normsTensorC1 = zeros(Float64, Lx)
        newTensorsC4_sF = Vector{TensorMap}(undef, Lx)
        newTensorsT4_sF = Vector{TensorMap}(undef, Lx)
        newTensorsC1_sF = Vector{TensorMap}(undef, Lx)
        for iterX = 1 : Lx

            # select pepsTensor, envTensors_gS and envTensors_sF
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_gS = environmentTensors_gS[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_sF = environmentTensors_sF[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            #------------------------

            # projector(s) for newC4
            truncationProjectorT = projectorsL[pIndex(iterX - 0, Lx)][2]

            # newC4 for envTensors_gS
            @tensor newC4[(); -1 -2] := envTensors_gS[7][2, 1] * envTensors_gS[6][1, 3, 4, -2] * truncationProjectorT[2, 3, 4, -1]
            newTensorsC4_gS[iterX], normsTensorC4[iterX] = postProcessEnvTensor_sF(newC4)

            # newC4 for envTensors_sF
            @tensor tmp1[(); -1 -2] := envTensors_gS[6][1, 3, 4, -2] * envTensors_sF[7][2, 1] * truncationProjectorT[2, 3, 4, -1]
            @tensor tmp2[(); -1 -2] := envTensors_sF[6][1, 3, 4, -2] * envTensors_gS[7][2, 1] * truncationProjectorT[2, 3, 4, -1]
            newC4 = (tmp1 + tmp2) * globalPhase_L
            newTensorsC4_sF[iterX] = newC4 / (2 * normsTensorC4[iterX])

            #------------------------

            # projector(s) for newT4
            truncationProjectorB = projectorsL[pIndex(iterX - 0, Lx)][1]
            truncationProjectorT = projectorsL[pIndex(iterX - 1, Lx)][2]

            # newT4 for envTensors_gS
            @tensor newT4[-1; -2 -3 -4] := truncationProjectorB[-1, 7, 8, 9] * envTensors_gS[8][7, 1, 2, 4] * pepsTensor[2, 8, 6, 3, -3] * conj(pepsTensor[4, 9, 6, 5, -4]) * truncationProjectorT[1, 3, 5, -2]
            newTensorsT4_gS[iterX], normsTensorT4[iterX] = postProcessEnvTensor_sF(newT4)

            # newT4 for envTensors_sF
            @tensor newT4[-1; -2 -3 -4] := truncationProjectorB[-1, 7, 8, 9] * envTensors_sF[8][7, 1, 2, 4] * pepsTensor[2, 8, 6, 3, -3] * conj(pepsTensor[4, 9, 6, 5, -4]) * truncationProjectorT[1, 3, 5, -2]
            for (basisSiteIdx, basisVector) = enumerate(basisVectors)
                basisOperator = getSpinOperatorFullBasis(basisSiteIdx, spinComponents[1], basisSize = length(basisVectors))
                @tensor tmp[-1; -2 -3 -4] := truncationProjectorB[-1, 8, 9, 10] * envTensors_gS[8][8, 2, 3, 5] * pepsTensor[3, 9, 1, 4, -3] * basisOperator[7, 1] * conj(pepsTensor[5, 10, 7, 6, -4]) * truncationProjectorT[2, 4, 6, -2]
                newT4 += (tmp * exp(+1im * dot(momentumVec, basisVector)))
            end
            newT4 *= globalPhase_L
            newTensorsT4_sF[iterX] = newT4 / ((1 + length(basisVectors)) * normsTensorT4[iterX])

            #------------------------

            # projector(s) for newC1
            truncationProjectorB = projectorsL[pIndex(iterX - 1, Lx)][1]

            # newC1 for envTensors_gS
            @tensor newC1[-1; -2] := envTensors_gS[1][2, 1] * envTensors_gS[2][1, 3, 4, -2] * truncationProjectorB[-1, 2, 3, 4]
            newTensorsC1_gS[iterX], normsTensorC1[iterX] = postProcessEnvTensor_sF(newC1)

            # newC1 for envTensors_sF
            @tensor tmp1[-1; -2] := envTensors_sF[1][2, 1] * envTensors_gS[2][1, 3, 4, -2] * truncationProjectorB[-1, 2, 3, 4]
            @tensor tmp2[-1; -2] := envTensors_gS[1][2, 1] * envTensors_sF[2][1, 3, 4, -2] * truncationProjectorB[-1, 2, 3, 4]
            newC1 = (tmp1 + tmp2) * globalPhase_L
            newTensorsC1_sF[iterX] = newC1 / (2 * normsTensorC1[iterX])

        end

        #--------------------------------------------------------

        # replace tensors for environmentTensors_gS
        for iterX = 1 : Lx
            environmentTensors_gS[unitCell[iterX, pIndex(iterY + 1, Ly)]][1] = newTensorsC1_gS[iterX]
            environmentTensors_gS[unitCell[iterX, pIndex(iterY + 1, Ly)]][8] = newTensorsT4_gS[iterX]
            environmentTensors_gS[unitCell[iterX, pIndex(iterY + 1, Ly)]][7] = newTensorsC4_gS[iterX]
        end

        # replace tensors for environmentTensors_sF
        for iterX = 1 : Lx
            environmentTensors_sF[unitCell[iterX, pIndex(iterY + 1, Ly)]][1] = newTensorsC1_sF[iterX]
            environmentTensors_sF[unitCell[iterX, pIndex(iterY + 1, Ly)]][8] = newTensorsT4_sF[iterX]
            environmentTensors_sF[unitCell[iterX, pIndex(iterY + 1, Ly)]][7] = newTensorsC4_sF[iterX]
        end

    end

    # function return
    return environmentTensors_gS, environmentTensors_sF;

end

function absorptionR_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors::Vector{Vector{Float64}}, globalPhase_R::ComplexF64, environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the right environment """

    Lx, Ly = size(unitCell)

    # loop over all columns
    for iterY = iterateCols(unitCell, reverseIterator = true, onlyUnique = true)

        # compute all projectors for the absorption of one column
        projectorsR = Vector{Vector{TensorMap}}(undef, Lx)
        for iterX = 1 : Lx
            unitCellView = [unitCell[pIndex(iterX + 0, Lx), pIndex(iterY - 1, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] ; unitCell[pIndex(iterX + 1, Lx), pIndex(iterY - 1, Ly)] unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 0, Ly)]]
            projectorsR[iterX] = calculateProjectorsR(pepsTensors[unitCellView], environmentTensors_gS[unitCellView], chiE, truncBelowE)
        end

        #--------------------------------------------------------

        # perform absorption of tensors for environmentTensors_gS and environmentTensors_sF
        newTensorsC2_gS = Vector{TensorMap}(undef, Lx)
        newTensorsT2_gS = Vector{TensorMap}(undef, Lx)
        newTensorsC3_gS = Vector{TensorMap}(undef, Lx)
        normsTensorC2 = zeros(Float64, Lx)
        normsTensorT2 = zeros(Float64, Lx)
        normsTensorC3 = zeros(Float64, Lx)
        newTensorsC2_sF = Vector{TensorMap}(undef, Lx)
        newTensorsT2_sF = Vector{TensorMap}(undef, Lx)
        newTensorsC3_sF = Vector{TensorMap}(undef, Lx)
        for iterX = 1 : Lx

            # select pepsTensor, envTensors_gS and envTensors_sF
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_gS = environmentTensors_gS[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_sF = environmentTensors_sF[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            #------------------------

            # projector(s) for newC2
            truncationProjectorB = projectorsR[pIndex(iterX - 1, Lx)][1]

            # newC2 for envTensors_gS
            @tensor newC2[-1 -2; ()] := envTensors_gS[2][-1, 2, 3, 1] * envTensors_gS[3][1, 4] * truncationProjectorB[-2, 2, 3, 4]
            newTensorsC2_gS[iterX], normsTensorC2[iterX] = postProcessEnvTensor_sF(newC2)

            # newC2 for envTensors_sF
            @tensor tmp1[-1 -2; ()] := envTensors_gS[2][-1, 2, 3, 1] * envTensors_sF[3][1, 4] * truncationProjectorB[-2, 2, 3, 4]
            @tensor tmp2[-1 -2; ()] := envTensors_sF[2][-1, 2, 3, 1] * envTensors_gS[3][1, 4] * truncationProjectorB[-2, 2, 3, 4]
            newC2 = (tmp1 + tmp2) * globalPhase_R
            newTensorsC2_sF[iterX] = newC2 / (2 * normsTensorC2[iterX])

            #------------------------

            # projector(s) for newT2
            truncationProjectorT = projectorsR[pIndex(iterX - 1, Lx)][2]
            truncationProjectorB = projectorsR[pIndex(iterX - 0, Lx)][1]

            # newT2 for envTensors_gS
            @tensor newT2[-1 -2 -3; -4] := truncationProjectorB[-3, 7, 8, 9] * pepsTensor[-1, 7, 6, 2, 3] * conj(pepsTensor[-2, 8, 6, 4, 5]) * envTensors_gS[4][3, 5, 9, 1] * truncationProjectorT[2, 4, 1, -4]
            newTensorsT2_gS[iterX], normsTensorT2[iterX] = postProcessEnvTensor_sF(newT2)

            # newT2 for envTensors_sF
            @tensor newT2[-1 -2 -3; -4] := truncationProjectorB[-3, 7, 8, 9] * pepsTensor[-1, 7, 6, 2, 3] * conj(pepsTensor[-2, 8, 6, 4, 5]) * envTensors_sF[4][3, 5, 9, 1] * truncationProjectorT[2, 4, 1, -4]
            for (basisSiteIdx, basisVector) = enumerate(basisVectors)
                basisOperator = getSpinOperatorFullBasis(basisSiteIdx, spinComponents[1], basisSize = length(basisVectors))
                @tensor tmp[-1 -2 -3; -4] := truncationProjectorB[-3, 8, 9, 10] * pepsTensor[-1, 8, 1, 3, 4] * basisOperator[7, 1] * conj(pepsTensor[-2, 9, 7, 5, 6]) * envTensors_gS[4][4, 6, 10, 2] * truncationProjectorT[3, 5, 2, -4]
                newT2 += (tmp * exp(+1im * dot(momentumVec, basisVector)))
            end
            newT2 *= globalPhase_R
            newTensorsT2_sF[iterX] = newT2 / ((1 + length(basisVectors)) * normsTensorT2[iterX])

            #------------------------

            # projector(s) for newC3
            truncationProjectorT = projectorsR[pIndex(iterX - 0, Lx)][2]

            # newC3 for envTensors_gS
            @tensor newC3[-1; -2] := envTensors_gS[6][-1, 2, 3, 1] * envTensors_gS[5][1, 4] * truncationProjectorT[2, 3, 4, -2]
            newTensorsC3_gS[iterX], normsTensorC3[iterX] = postProcessEnvTensor_sF(newC3)

            # newC3 for envTensors_sF
            @tensor tmp1[-1; -2] := envTensors_sF[5][1, 4] * envTensors_gS[6][-1, 2, 3, 1] * truncationProjectorT[2, 3, 4, -2]
            @tensor tmp2[-1; -2] := envTensors_gS[5][1, 4] * envTensors_sF[6][-1, 2, 3, 1] * truncationProjectorT[2, 3, 4, -2]
            newC3 = (tmp1 + tmp2) * globalPhase_R
            newTensorsC3_sF[iterX] = newC3 / (2 * normsTensorC3[iterX])

        end

        # replace tensors for environmentTensors_gS
        for iterX = 1 : Lx
            environmentTensors_gS[unitCell[iterX, pIndex(iterY - 1, Ly)]][3] = newTensorsC2_gS[iterX]
            environmentTensors_gS[unitCell[iterX, pIndex(iterY - 1, Ly)]][4] = newTensorsT2_gS[iterX]
            environmentTensors_gS[unitCell[iterX, pIndex(iterY - 1, Ly)]][5] = newTensorsC3_gS[iterX]
        end

        # replace tensors for environmentTensors_sF
        for iterX = 1 : Lx
            environmentTensors_sF[unitCell[iterX, pIndex(iterY - 1, Ly)]][3] = newTensorsC2_sF[iterX]
            environmentTensors_sF[unitCell[iterX, pIndex(iterY - 1, Ly)]][4] = newTensorsT2_sF[iterX]
            environmentTensors_sF[unitCell[iterX, pIndex(iterY - 1, Ly)]][5] = newTensorsC3_sF[iterX]
        end

    end

    # function return
    return environmentTensors_gS, environmentTensors_sF;

end

function absorptionB_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors::Vector{Vector{Float64}}, globalPhase_B::ComplexF64, environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the bottom environment """

    Lx, Ly = size(unitCell)

    # loop over all rows
    for iterX = iterateRows(unitCell, reverseIterator = true, onlyUnique = true)

        # compute all projectors for the absorption of one row
        projectorsB = Vector{Vector{TensorMap}}(undef, Ly)
        for iterY = 1 : Ly
            unitCellView = [unitCell[pIndex(iterX - 1, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX - 1, Lx), pIndex(iterY + 1, Ly)] ; unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 1, Ly)]]
            projectorsB[iterY] = calculateProjectorsB(pepsTensors[unitCellView], environmentTensors_gS[unitCellView], chiE, truncBelowE)
        end

        #--------------------------------------------------------

        # perform absorption of tensors for environmentTensors_gS and environmentTensors_sF
        newTensorsC3_gS = Vector{TensorMap}(undef, Ly)
        newTensorsT3_gS = Vector{TensorMap}(undef, Ly)
        newTensorsC4_gS = Vector{TensorMap}(undef, Ly)
        normsTensorC3 = zeros(Float64, Ly)
        normsTensorT3 = zeros(Float64, Ly)
        normsTensorC4 = zeros(Float64, Ly)
        newTensorsC3_sF = Vector{TensorMap}(undef, Ly)
        newTensorsT3_sF = Vector{TensorMap}(undef, Ly)
        newTensorsC4_sF = Vector{TensorMap}(undef, Ly)
        for iterY = 1 : Ly

            # select pepsTensor, envTensors_gS and envTensors_sF
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_gS = environmentTensors_gS[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_sF = environmentTensors_sF[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            #------------------------

            # projector(s) for newC3
            truncationProjectorL = projectorsB[pIndex(iterY - 0, Ly)][1]

            # newC3 for envTensors_gS
            @tensor newC3[-1; -2] := truncationProjectorL[-1, 2, 3, 4] * envTensors_gS[4][2, 3, 1, -2] * envTensors_gS[5][4, 1]
            newTensorsC3_gS[iterY], normsTensorC3[iterY] = postProcessEnvTensor_sF(newC3)

            # newC3 for envTensors_sF
            @tensor tmp1[-1; -2] := truncationProjectorL[-1, 2, 3, 4] * envTensors_gS[4][2, 3, 1, -2] * envTensors_sF[5][4, 1]
            @tensor tmp2[-1; -2] := truncationProjectorL[-1, 2, 3, 4] * envTensors_sF[4][2, 3, 1, -2] * envTensors_gS[5][4, 1]
            newC3 = (tmp1 + tmp2) * globalPhase_B
            newTensorsC3_sF[iterY] = newC3 / (2 * normsTensorC3[iterY])

            #------------------------

            # projector(s) for newT3
            truncationProjectorR = projectorsB[pIndex(iterY - 0, Ly)][2]
            truncationProjectorL = projectorsB[pIndex(iterY - 1, Ly)][1]

            # newT3 for envTensors_gS
            @tensor newT3[-1; -2 -3 -4] := truncationProjectorL[-1, 7, 8, 9] * envTensors_gS[6][9, 2, 4, 1] * pepsTensor[7, 2, 6, -2, 3] * conj(pepsTensor[8, 4, 6, -3, 5]) * truncationProjectorR[3, 5, 1, -4]
            newTensorsT3_gS[iterY], normsTensorT3[iterY] = postProcessEnvTensor_sF(newT3)

            # newT3 for envTensors_sF
            @tensor newT3[-1; -2 -3 -4] := truncationProjectorL[-1, 7, 8, 9] * envTensors_sF[6][9, 2, 4, 1] * pepsTensor[7, 2, 6, -2, 3] * conj(pepsTensor[8, 4, 6, -3, 5]) * truncationProjectorR[3, 5, 1, -4]
            for (basisSiteIdx, basisVector) = enumerate(basisVectors)
                basisOperator = getSpinOperatorFullBasis(basisSiteIdx, spinComponents[1], basisSize = length(basisVectors))
                @tensor tmp[-1; -2 -3 -4] := truncationProjectorL[-1, 8, 9, 10] * envTensors_gS[6][10, 3, 5, 2] * pepsTensor[8, 3, 1, -2, 4] * basisOperator[7, 1] * conj(pepsTensor[9, 5, 7, -3, 6]) * truncationProjectorR[4, 6, 2, -4]
                newT3 += (tmp * exp(+1im * dot(momentumVec, basisVector)))
            end
            newT3 *= globalPhase_B
            newTensorsT3_sF[iterY] = newT3 / ((1 + length(basisVectors)) * normsTensorT3[iterY])

            #------------------------

            # projector(s) for newC4
            truncationProjectorR = projectorsB[pIndex(iterY - 1, Ly)][2]

            # newC4 for envTensors_gS
            @tensor newC4[(); -1 -2] := envTensors_gS[7][1, 4] * envTensors_gS[8][1, -1, 2, 3] * truncationProjectorR[2, 3, 4, -2]
            newTensorsC4_gS[iterY], normsTensorC4[iterY] = postProcessEnvTensor_sF(newC4)

            # newC4 for envTensors_sF
            @tensor tmp1[(); -1 -2] := envTensors_sF[7][1, 4] * envTensors_gS[8][1, -1, 2, 3] * truncationProjectorR[2, 3, 4, -2]
            @tensor tmp2[(); -1 -2] := envTensors_gS[7][1, 4] * envTensors_sF[8][1, -1, 2, 3] * truncationProjectorR[2, 3, 4, -2]
            newC4 = (tmp1 + tmp2) * globalPhase_B
            newTensorsC4_sF[iterY] = newC4 / (2 * normsTensorC4[iterY])

        end

        # replace tensors for environmentTensors_gS
        for iterY = 1 : Ly
            environmentTensors_gS[unitCell[pIndex(iterX - 1, Lx), iterY]][5] = newTensorsC3_gS[iterY]
            environmentTensors_gS[unitCell[pIndex(iterX - 1, Lx), iterY]][6] = newTensorsT3_gS[iterY]
            environmentTensors_gS[unitCell[pIndex(iterX - 1, Lx), iterY]][7] = newTensorsC4_gS[iterY]
        end

        # replace tensors for environmentTensors_sF
        for iterY = 1 : Ly
            environmentTensors_sF[unitCell[pIndex(iterX - 1, Lx), iterY]][5] = newTensorsC3_sF[iterY]
            environmentTensors_sF[unitCell[pIndex(iterX - 1, Lx), iterY]][6] = newTensorsT3_sF[iterY]
            environmentTensors_sF[unitCell[pIndex(iterX - 1, Lx), iterY]][7] = newTensorsC4_sF[iterY]
        end

    end

    # function return
    return environmentTensors_gS, environmentTensors_sF;

end

function absorptionT_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors::Vector{Vector{Float64}}, globalPhase_T::ComplexF64, environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    """ computes projectors and absorbs full unit cell into the top environment """

    Lx, Ly = size(unitCell)

    # loop over all rows
    for iterX = iterateRows(unitCell, reverseIterator = false, onlyUnique = true)

        # compute all projectors for the absorption of one row
        projectorsT = Vector{Vector{TensorMap}}(undef, Ly)
        for iterY = 1 : Ly
            unitCellView = [unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 0, Lx), pIndex(iterY + 1, Ly)] ; unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 0, Ly)] unitCell[pIndex(iterX + 1, Lx), pIndex(iterY + 1, Ly)]]
            projectorsT[iterY] = calculateProjectorsT(pepsTensors[unitCellView], environmentTensors_gS[unitCellView], chiE, truncBelowE)
        end

        #--------------------------------------------------------

        # perform absorption of tensors for environmentTensors_gS and environmentTensors_sF
        newTensorsC1_gS = Vector{TensorMap}(undef, Ly)
        newTensorsT1_gS = Vector{TensorMap}(undef, Ly)
        newTensorsC2_gS = Vector{TensorMap}(undef, Ly)
        normsTensorC1 = zeros(Float64, Ly)
        normsTensorT1 = zeros(Float64, Ly)
        normsTensorC2 = zeros(Float64, Ly)
        newTensorsC1_sF = Vector{TensorMap}(undef, Ly)
        newTensorsT1_sF = Vector{TensorMap}(undef, Ly)
        newTensorsC2_sF = Vector{TensorMap}(undef, Ly)
        for iterY = 1 : Ly

            # select pepsTensor, envTensors_gS and envTensors_sF
            pepsTensor = pepsTensors[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_gS = environmentTensors_gS[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]
            envTensors_sF = environmentTensors_sF[unitCell[pIndex(iterX, Lx), pIndex(iterY, Ly)]]

            #------------------------

            # projector(s) for newC1
            truncationProjectorR = projectorsT[pIndex(iterY - 1, Ly)][2]

            # newC1 for envTensors_gS
            @tensor newC1[-1; -2] := envTensors_gS[8][-1, 1, 3, 4] * envTensors_gS[1][1, 2] * truncationProjectorR[2, 3, 4, -2]
            newTensorsC1_gS[iterY], normsTensorC1[iterY] = postProcessEnvTensor_sF(newC1)

            # newC1 for envTensors_sF
            @tensor tmp1[-1; -2] := envTensors_gS[8][-1, 1, 3, 4] * envTensors_sF[1][1, 2] * truncationProjectorR[2, 3, 4, -2]
            @tensor tmp2[-1; -2] := envTensors_sF[8][-1, 1, 3, 4] * envTensors_gS[1][1, 2] * truncationProjectorR[2, 3, 4, -2]
            newC1 = (tmp1 + tmp2) * globalPhase_T
            newTensorsC1_sF[iterY] = newC1 / (2 * normsTensorC1[iterY])

            #------------------------

            # projector(s) for newT1
            truncationProjectorL = projectorsT[pIndex(iterY - 1, Ly)][1]
            truncationProjectorR = projectorsT[pIndex(iterY - 0, Ly)][2]

            # newT1 for environmentTensors_gS
            @tensor newT1[-1 -2 -3; -4] := truncationProjectorL[-1, 7, 8, 9] * envTensors_gS[2][7, 2, 4, 1] * pepsTensor[8, -2, 6, 2, 3] * conj(pepsTensor[9, -3, 6, 4, 5]) * truncationProjectorR[1, 3, 5, -4]
            newTensorsT1_gS[iterY], normsTensorT1[iterY] = postProcessEnvTensor_sF(newT1)

            # newT1 for environmentTensors_sF
            @tensor newT1[-1 -2 -3; -4] := truncationProjectorL[-1, 7, 8, 9] * envTensors_sF[2][7, 2, 4, 1] * pepsTensor[8, -2, 6, 2, 3] * conj(pepsTensor[9, -3, 6, 4, 5]) * truncationProjectorR[1, 3, 5, -4]
            for (basisSiteIdx, basisVector) = enumerate(basisVectors)
                basisOperator = getSpinOperatorFullBasis(basisSiteIdx, spinComponents[1], basisSize = length(basisVectors))
                @tensor tmp[-1 -2 -3; -4] := truncationProjectorL[-1, 8, 9, 10] * envTensors_gS[2][8, 3, 5, 2] * pepsTensor[9, -2, 1, 3, 4] * basisOperator[7, 1] * conj(pepsTensor[10, -3, 7, 5, 6]) * truncationProjectorR[2, 4, 6, -4]
                newT1 += (tmp * exp(+1im * dot(momentumVec, basisVector)))
            end
            newT1 *= globalPhase_T
            newTensorsT1_sF[iterY] = newT1 / ((1 + length(basisVectors)) * normsTensorT1[iterY])

            #------------------------

            # projector(s) for newC2
            truncationProjectorL = projectorsT[pIndex(iterY - 0, Ly)][1]

            # newC2 for environmentTensors_gS
            @tensor newC2[-1 -2; ()] := envTensors_gS[3][2, 1] * envTensors_gS[4][3, 4, -2, 1] * truncationProjectorL[-1, 2, 3, 4]
            newTensorsC2_gS[iterY], normsTensorC2[iterY] = postProcessEnvTensor_sF(newC2)

            # newC2 for environmentTensors_sF
            @tensor tmp1[-1 -2; ()] := envTensors_sF[3][2, 1] * envTensors_gS[4][3, 4, -2, 1] * truncationProjectorL[-1, 2, 3, 4]
            @tensor tmp2[-1 -2; ()] := envTensors_gS[3][2, 1] * envTensors_sF[4][3, 4, -2, 1] * truncationProjectorL[-1, 2, 3, 4]
            newC2 = (tmp1 + tmp2) * globalPhase_T
            newTensorsC2_sF[iterY] = newC2 / (2 * normsTensorC2[iterY])

        end

        # replace tensors for environmentTensors_gS
        for iterY = 1 : Ly
            environmentTensors_gS[unitCell[pIndex(iterX + 1, Lx), iterY]][1] = newTensorsC1_gS[iterY]
            environmentTensors_gS[unitCell[pIndex(iterX + 1, Lx), iterY]][2] = newTensorsT1_gS[iterY]
            environmentTensors_gS[unitCell[pIndex(iterX + 1, Lx), iterY]][3] = newTensorsC2_gS[iterY]
        end

        # replace tensors for environmentTensors_sF
        for iterY = 1 : Ly
            environmentTensors_sF[unitCell[pIndex(iterX + 1, Lx), iterY]][1] = newTensorsC1_sF[iterY]
            environmentTensors_sF[unitCell[pIndex(iterX + 1, Lx), iterY]][2] = newTensorsT1_sF[iterY]
            environmentTensors_sF[unitCell[pIndex(iterX + 1, Lx), iterY]][3] = newTensorsC2_sF[iterY]
        end

    end

    # function return
    return environmentTensors_gS, environmentTensors_sF

end

function doAbsorptionStep_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors::Vector{Vector{Float64}}, globalPhases::Vector{ComplexF64}, environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    environmentTensors_gS, environmentTensors_sF = absorptionL_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors, globalPhases[1], environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    environmentTensors_gS, environmentTensors_sF = absorptionT_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors, globalPhases[2], environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    environmentTensors_gS, environmentTensors_sF = absorptionR_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors, globalPhases[3], environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    environmentTensors_gS, environmentTensors_sF = absorptionB_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors, globalPhases[4], environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)
    return environmentTensors_gS, environmentTensors_sF
end

function phaseCTMRG_sF(unitCell::Matrix{Int64}, pepsTensors, momentumVec::Vector{Float64}, spinComponents::Vector{Int64}, latticeVectors::Vector{Vector{Float64}}, basisVectors::Vector{Vector{Float64}}, chiE::Int64; maxSteps::Int64 = 100, truncBelowE::Float64 = 1e-8, convTolE::Float64 = 1e-6, verbosePrint::Bool = false, printSummary::Bool = true)

    # initialize environmentTensors for the ground state (no phase factors included)
    environmentTensors_gS = initializeEnvironments(pepsTensors)

    # initialize environmentTensors for the structure factor (phase factors included)
    environmentTensors_sF = initializeEnvironments(pepsTensors)

    # construct globalPhases for the four absorption directions [L, T, R, B]
    globalPhases = exp.([
        -1im * dot(momentumVec, latticeVectors[2]), 
        -1im * dot(momentumVec, latticeVectors[1]),
        +1im * dot(momentumVec, latticeVectors[2]),
        +1im * dot(momentumVec, latticeVectors[1]),
    ])

    # initialize cornerSingularValues to check convergence
    oldCornerSingularVals_gS = zeros(Float64, length(pepsTensors), 4, chiE)
    oldCornerSingularVals_sF = zeros(Float64, length(pepsTensors), 4, chiE)

    # run CTMRG
    runCTMRG = true
    envLoopCounter = 1
    normSingularValues_gS = 1
    normSingularValues_sF = 1
    while runCTMRG

        # do full CTMRG absorption step
        environmentTensors_gS, environmentTensors_sF = doAbsorptionStep_sF(unitCell, pepsTensors, momentumVec, spinComponents, basisVectors, globalPhases, environmentTensors_gS, environmentTensors_sF, chiE, truncBelowE)

        # compute corner SVDs and check convergence
        newCornerSingularVals_gS = calculateCornerSVDs(environmentTensors_gS, chiE)
        newCornerSingularVals_sF = calculateCornerSVDs(environmentTensors_sF, chiE)
        normSingularValues_gS = norm(newCornerSingularVals_gS - oldCornerSingularVals_gS)
        normSingularValues_sF = norm(newCornerSingularVals_sF - oldCornerSingularVals_sF)
        if ((normSingularValues_gS < convTolE) && (normSingularValues_sF < convTolE)) || (envLoopCounter == maxSteps)
            runCTMRG = false
        end
        oldCornerSingularVals_gS = newCornerSingularVals_gS
        oldCornerSingularVals_sF = newCornerSingularVals_sF

        # print CTMRG convergence info
        verbosePrint && @printf("CTMRG Step %03d, norm SVs: %0.6e (gS), %0.6e (sF)\n", envLoopCounter, normSingularValues_gS, normSingularValues_sF)

        # increase envLoopCounter
        envLoopCounter += 1

    end

    # return number of CTM steps necessary to achieve convergence
    numE = envLoopCounter - 1

    # print CTMRG convergence info
    printSummary && @printf("CTMRG conv. in %d steps, norm of SVs: %0.6e (gS), %0.6e (sF)\n", numE, normSingularValues_gS, normSingularValues_sF)

    # function return
    return environmentTensors_gS, environmentTensors_sF

end

function getPhaseEnvironmentTensors(envIdx, envTensors_gS, envTensors_sF)
    phaseEnvTensors = Vector{TensorMap}(undef, length(envTensors_gS))
    for (idx, envTensor) in enumerate(envTensors_gS)
        phaseEnvTensors[idx] = envTensor
    end
    phaseEnvTensors[envIdx] = envTensors_sF[envIdx]
    return phaseEnvTensors
end

function computeStructureFactor(unitCell::Matrix{Int64}, pepsTensors, momentumVec::Vector{Float64}, spinComponents::Vector{Int64}, latticeVectors, basisVectors, chiE, convTolE, truncBelowE::Float64 = 1e-8, verbosePrint::Bool = false)

    # initialize structureFactorValue
    structureFactorValue_C = 0.0 + 0.0im
    structureFactorValue_D = 0.0 + 0.0im

    # run phaseCTMRG
    environmentTensors_gS, environmentTensors_sF = phaseCTMRG_sF(unitCell, pepsTensors, momentumVec, spinComponents, latticeVectors, basisVectors, chiE, convTolE = convTolE, truncBelowE = truncBelowE, verbosePrint = verbosePrint)

    # compute single-site norm for envTensors_gS
    oneSiteRDM_gS = computeOneSiteRDM_iPEPS(pepsTensors[1], environmentTensors_gS[1])
    oneSiteNorm = real(tr(oneSiteRDM_gS))
    
    # loop over all basis sites in the unit cell
    for (basisSiteIdxA, basisVectorA) = enumerate(basisVectors)

        # construct operatorA for full six-site basis and compute its expectation value
        basisOperatorA = getSpinOperatorFullBasis(basisSiteIdxA, spinComponents[2], basisSize = length(basisVectors))
        expValBasisOpA = tr(basisOperatorA * oneSiteRDM_gS) / oneSiteNorm

        # compute local contributions
        storeLocalTerms = zeros(ComplexF64, 0, 2)
        for (basisSiteIdxB, basisVectorB) = enumerate(basisVectors)

            if basisSiteIdxA != basisSiteIdxB
            
                # construct operatorB for full six-site basis and compute its expectation value
                basisOperatorB = getSpinOperatorFullBasis(basisSiteIdxB, spinComponents[1], basisSize = length(basisVectors))
                expValBasisOpB = tr(basisOperatorB * oneSiteRDM_gS) / oneSiteNorm

                # construct operatorA * operatorB for the full six-site basis and compute its expectation value
                fullBasisOperator = basisOperatorA * basisOperatorB
                expValBasisOpAB = tr(fullBasisOperator * oneSiteRDM_gS) / oneSiteNorm

                # display([expValBasisOpAB expValBasisOpA expValBasisOpB expValBasisOpA * expValBasisOpB])

                # compute connected and disconnected part of the correlation
                valueSF_C = exp(+1im * dot(momentumVec, basisVectorB .- basisVectorA)) * expValBasisOpAB
                valueSF_D = exp(+1im * dot(momentumVec, basisVectorB .- basisVectorA)) * (expValBasisOpAB - expValBasisOpA * expValBasisOpB)

                # store structure factors
                storeLocalTerms = vcat(storeLocalTerms, [valueSF_C valueSF_D])
                structureFactorValue_C += valueSF_C
                structureFactorValue_D += valueSF_D

            end
            
        end

        # compute non-local contributions
        envContributions = zeros(ComplexF64, 8, 2)
        for envIdx = 1 : 8

            # get environment with only one tensor that includes operators and phases
            phaseEnvTensors = getPhaseEnvironmentTensors(envIdx, environmentTensors_gS[1], environmentTensors_sF[1])

            # compute reduced density matrix using the regular PEPS tensor
            oneSiteRDM_sF = computeOneSiteRDM_iPEPS(pepsTensors[1], phaseEnvTensors)
            expValBasisOpB = tr(oneSiteRDM_sF) / oneSiteNorm

            # compute expectation value for operatorA * operatorB
            expValBasisOpAB = tr(basisOperatorA * oneSiteRDM_sF) / oneSiteNorm

            # compute connected and disconnected part of the correlation
            valueSF_C = exp(-1im * dot(momentumVec, basisVectorA)) * expValBasisOpAB
            valueSF_D = exp(-1im * dot(momentumVec, basisVectorA)) * (expValBasisOpAB - expValBasisOpA * expValBasisOpB)

            # store structure factors
            envContributions[envIdx, :] = [valueSF_C valueSF_D]
            structureFactorValue_C += valueSF_C
            structureFactorValue_D += valueSF_D

        end

    end

    # normalize sublattice magnetization to number of spins in the basis
    structureFactorValue_C /= length(basisVectors)
    structureFactorValue_D /= length(basisVectors)

    # function return
    return structureFactorValue_C, structureFactorValue_D

end