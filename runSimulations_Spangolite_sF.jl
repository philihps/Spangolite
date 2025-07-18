#!/usr/bin/env julia

# clear console
Base.run(`clear`)

# include packages
using JLD
using LinearAlgebra
using Plots
using Printf
using Statistics
using TensorKit

# include functions
include("computeStructureFactorInfPEPS.jl")
include("computeExpectationValues.jl")

let

    # -----------------------------------------------------------

    # set momentum [kx, ky] in Brillouin zone
    edgeBZ = 3.00;
    stepBZ = 0.10;

    #   2 | 1
    # ----|----
    #   3 | 4

    # select quadrant
    selectQuadrant = 0;

    # construct Brillouin zone for one quadrant
    if selectQuadrant == 0
        brillouinX = π * collect(-edgeBZ : stepBZ : +edgeBZ)
        brillouinY = π * collect(-edgeBZ : stepBZ : +edgeBZ)
    elseif selectQuadrant == 1
        brillouinX = π * collect(+0.0 : stepBZ : +edgeBZ)
        brillouinY = π * collect(+0.0 : stepBZ : +edgeBZ)
    elseif selectQuadrant == 2
        brillouinX = π * collect(-edgeBZ : stepBZ : -stepBZ)
        brillouinY = π * collect(+0 : stepBZ : +edgeBZ)
    elseif selectQuadrant == 3
        brillouinX = π * collect(-edgeBZ : stepBZ : -stepBZ)
        brillouinY = π * collect(-edgeBZ : stepBZ : -stepBZ)
    elseif selectQuadrant == 4
        brillouinX = π * collect(+0 : stepBZ : +edgeBZ)
        brillouinY = π * collect(-edgeBZ : stepBZ : -stepBZ)
    end

    # -----------------------------------------------------------

    # set lattice spacing a and lattice vectors [a1, a2]
    a = 1.0;
    latticeVectors = a .* [
        sqrt(7) / 2 * [-1.0, -sqrt(3)], 
        sqrt(7) / 1 * [+1.0, +0.0]
    ]

    # set basis vectors for six-site basis of the maple-leaf lattice
    basisVectors = a .* [
        [0.0, 0.0], 
        1 / (2 * sqrt(7)) * [+1.0, -3 * sqrt(3)], 
        1 / (1 * sqrt(7)) * [+1.0, -3 * sqrt(3)], 
        1 / (1 * sqrt(7)) * [+3.0, -2 * sqrt(3)], 
        1 / (2 * sqrt(7)) * [+5.0, -1 * sqrt(3)], 
        1 / (1 * sqrt(7)) * [+5.0, -1 * sqrt(3)]
    ]

    # -----------------------------------------------------------

    # set lattice and model
    latticeName = "mapleLeafLattice"
    modelName = "Spangolite"

    # Hamiltonian couplings in K
    J1 = +95.0367
    J2 = -23.9085
    J3 = -24.1591
    J4 = +28.2218
    J5 = +41.5641

    # set magnetic fields
    hamCouplingsHz = collect(0.0 : 0.1 : 0.0)

    # set physical system
    physicalSpin = 1/2
    setSym = "SU0"

    # set unit cell on the honeycomb lattice
    Lx = 1
    Ly = 1
    if Lx == 1 && Ly == 1
        unitCell = [1 1 ; 1 1]
    elseif Lx == 1 && Ly == 2
        unitCell = [1 2 ; 2 1]
    elseif Lx == 1 && Ly == 3
        unitCell = [1 2 3 ; 2 3 1 ; 3 1 2]
    else
        unitCell = reshape(collect(1 : Lx * Ly), Lx, Ly)
    end

    # set bulk bond dimension and convergence
    bondDimsB = collect(1 : 1)
    bondDimsB = [1]
    
    # set convergence tolerances
    convTolE = 1e-6

    # set tensorUpdateFlag ("iPESS" or "iPEPS" simple update tensors)
    tensorUpdateFlag = "iPESS";

    # loop over all magneticFields
    for idxHz = eachindex(hamCouplingsHz)

        # get magnetic field value
        Hz = hamCouplingsHz[idxHz]
        modelParameters = Vector{Float64}([J1, J2, J3, J4, J5, Hz])

        # loop over all chiB
        for chiBIdx = eachindex(bondDimsB)
            
            # get chiB
            chiB = bondDimsB[chiBIdx]

            # determine bondDimsE
            if chiB <= 2
                bondDimsE = [chiB]
            else
                bondDimsE = chiB .* collect((chiB - 0) : (chiB - 0))
            end

            # loop over all chiE
            for chiEIdx = eachindex(bondDimsE)

                # get chiE
                chiE = bondDimsE[chiEIdx]

                # loop over all kx
                for (idx, kx) in enumerate(brillouinX)

                    # loop over all ky
                    for (idy, ky) in enumerate(brillouinY)

                        # construct modelDict
                        modelDict = Dict(
                            "latticeName" => latticeName,
                            "modelName" => modelName,
                            "physicalSpin" => physicalSpin,
                            "setSym" => setSym,
                            "Lx" => Lx,
                            "Ly" => Ly,
                            "unitCell" => unitCell,
                            "modelParameters" => modelParameters,
                            "chiB" => chiB,
                            "chiE" => chiE,
                            "convTolE" => convTolE,
                            "latticeVectors" => latticeVectors,
                            "basisVectors" => basisVectors,
                            "kx" => kx,
                            "ky" => ky
                        )

                        # set main directory path
                        mainDirPath = @sprintf("numFiles_%s/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", tensorUpdateFlag, latticeName, modelName, physicalSpin, setSym, Lx, Ly)

                        # set fileString
                        folderStringSimpleUpdate = mainDirPath * "/simpleUpdate/bondDim_" * string(chiB)
                        fileStringSimpleUpdate = @sprintf("%s/simpleUpdate_hamVars_", folderStringSimpleUpdate)
                        for varIdx = eachindex(modelParameters)
                            if varIdx < length(modelParameters)
                                fileStringSimpleUpdate *= @sprintf("%+0.2f_", modelParameters[varIdx])
                            else
                                fileStringSimpleUpdate *= @sprintf("%+0.3f_", modelParameters[varIdx])
                            end
                        end
                        fileStringSimpleUpdate *= @sprintf("chiB_%d.jld", chiB)

                        # compute structureFactor
                        if isfile(fileStringSimpleUpdate)
                            computeStructureFactorInfPEPS(modelDict, tensorUpdateFlag)
                        end

                    end
                    
                end

            end

        end

    end
    
end