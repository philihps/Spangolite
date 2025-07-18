function hosvd_iPESO(T, indBondDim, nodeType, doTruncation; truncBelow::Float64 = 1e-14)

    # initialize cell array for unitary matrices
    numberOfDims = 3
    listOfUnitaries = Vector{TensorMap}(undef, numberOfDims)
    listOfAdjointUnitaries = Vector{TensorMap}(undef, numberOfDims)
    listOfSingularValues = Vector{TensorMap}(undef, numberOfDims)
    discardedWeight = zeros(Float64, numberOfDims)

    if nodeType == -1
        
        # HOSVD for index 1
        idx = 1
        @tensor modT[-1 -2 -3; -4 -5 -6] := T[-1 -2 -3 1 2 3 4 5 6] * conj(T[-4 -5 -6 1 2 3 4 5 6])
        if doTruncation == 1
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncdim(indBondDim[1]), alg = TensorKit.SVD())
            # U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncbelow(truncBelow) & truncdim(indBondDim[1]), alg = TensorKit.SVD())
        else
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, alg = TensorKit.SVD())
        end
        S = sqrt(S)
        S /= norm(S)

        # store discardedWeight
        discardedWeight[idx] = sqrt(ϵ)

        # store variables
        listOfUnitaries[idx] = U
        listOfSingularValues[idx] = S
        listOfAdjointUnitaries[idx] = V


        # HOSVD for index 2
        idx = 2
        @tensor modT[-1 -2 -3; -4 -5 -6] := T[1 2 3 -4 -5 -6 4 5 6] * conj(T[1 2 3 -1 -2 -3 4 5 6])
        if doTruncation == 1
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncdim(indBondDim[2]), alg = TensorKit.SVD())
            # U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncbelow(truncBelow) & truncdim(indBondDim[2]), alg = TensorKit.SVD())
        else
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, alg = TensorKit.SVD())
        end
        S = sqrt(S)
        S /= norm(S)

        # store discardedWeight
        discardedWeight[idx] = sqrt(ϵ)

        # store variables
        listOfUnitaries[idx] = V
        listOfSingularValues[idx] = S
        listOfAdjointUnitaries[idx] = U


        # HOSVD for index 3
        idx = 3
        @tensor modT[-1 -2 -3; -4 -5 -6] := T[1 2 3 4 5 6 -4 -5 -6] * conj(T[1 2 3 4 5 6 -1 -2 -3])
        if doTruncation == 1
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncdim(indBondDim[3]), alg = TensorKit.SVD())
            # U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncbelow(truncBelow) & truncdim(indBondDim[3]), alg = TensorKit.SVD())
        else
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, alg = TensorKit.SVD())
        end
        S = sqrt(S)
        S /= norm(S)

        # store discardedWeight
        discardedWeight[idx] = sqrt(ϵ)

        # store variables
        listOfUnitaries[idx] = V
        listOfSingularValues[idx] = S
        listOfAdjointUnitaries[idx] = U


        # construct new coreTensor
        @tensor coreTensor[-1; -2 -3] := T[1 2 3 4 5 6 7 8 9] * listOfAdjointUnitaries[1][-1 1 2 3] * listOfAdjointUnitaries[2][4 5 6 -2] * listOfAdjointUnitaries[3][7 8 9 -3]
        coreTensor /= norm(coreTensor)

    elseif nodeType == +1


        # HOSVD for index 1
        idx = 1
        @tensor modT[-1 -2 -3; -4 -5 -6] := T[-1 -2 -3 1 2 3 4 5 6] * conj(T[-4 -5 -6 1 2 3 4 5 6])
        if doTruncation == 1
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncdim(indBondDim[1]), alg = TensorKit.SVD())
            # U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncbelow(truncBelow) & truncdim(indBondDim[1]), alg = TensorKit.SVD())
        else
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, alg = TensorKit.SVD())
        end
        S = sqrt(S)
        S /= norm(S)

        # store discardedWeight
        discardedWeight[idx] = sqrt(ϵ)

        # store variables
        listOfUnitaries[idx] = U
        listOfSingularValues[idx] = S
        listOfAdjointUnitaries[idx] = V


        # HOSVD for index 2
        idx = 2
        @tensor modT[-1 -2 -3; -4 -5 -6] := T[1 2 3 -1 -2 -3 4 5 6] * conj(T[1 2 3 -4 -5 -6 4 5 6])
        if doTruncation == 1
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncdim(indBondDim[2]), alg = TensorKit.SVD())
            # U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncbelow(truncBelow) & truncdim(indBondDim[2]), alg = TensorKit.SVD())
        else
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, alg = TensorKit.SVD())
        end
        S = sqrt(S)
        S /= norm(S)

        # store discardedWeight
        discardedWeight[idx] = sqrt(ϵ)

        # store variables
        listOfUnitaries[idx] = U
        listOfSingularValues[idx] = S
        listOfAdjointUnitaries[idx] = V


        # HOSVD for index 3
        idx = 3
        @tensor modT[-1 -2 -3; -4 -5 -6] := T[1 2 3 4 5 6 -4 -5 -6] * conj(T[1 2 3 4 5 6 -1 -2 -3])
        if doTruncation == 1
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncdim(indBondDim[3]), alg = TensorKit.SVD())
            # U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, trunc = truncbelow(truncBelow) & truncdim(indBondDim[3]), alg = TensorKit.SVD())
        else
            U, S, V, ϵ = tsvd(modT, (1, 2, 3), (4, 5, 6), p = 1, alg = TensorKit.SVD())
        end
        S = sqrt(S)
        S /= norm(S)

        # store discardedWeight
        discardedWeight[idx] = sqrt(ϵ)

        # store variables
        listOfUnitaries[idx] = V
        listOfSingularValues[idx] = S
        listOfAdjointUnitaries[idx] = U

        # construct new coreTensor
        @tensor coreTensor[-1 -2; -3] := T[1 2 3 4 5 6 7 8 9] * listOfAdjointUnitaries[1][-1 1 2 3] * listOfAdjointUnitaries[2][-2 4 5 6] * listOfAdjointUnitaries[3][7 8 9 -3]
        coreTensor /= norm(coreTensor)

    end

    return coreTensor, listOfUnitaries, listOfSingularValues, discardedWeight

end