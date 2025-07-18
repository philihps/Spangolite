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

# include project specific functions
include("simpleUpdate_iPESO.jl")
include("computeEnvironment.jl")
include("computeExpectationValues.jl")

let

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

    # set unit cell on the Kagome lattice
    Lx = 1
    Ly = 1
    if Lx == 1 && Ly == 2
        unitCell = [1 2 ; 2 1]
    elseif Lx == 1 && Ly == 3
        unitCell = [1 2 3 ; 2 3 1 ; 3 1 2]
    else
        unitCell = reshape(collect(1 : Lx * Ly), Lx, Ly)
    end

    # set bond dimensions
    bondDimsB = collect(8 : 1 : 8)

    # infinitesimal inverse temperature step and inverse temperatures
    δβ = 1e-3
    inverseTemperatures = δβ .* vcat(collect(1 : 1 : 99))
    # inverseTemperatures = δβ .* vcat(collect(1 : 1 : 99), collect(100 : 10 : 990), collect(1000 : 100 : 9900), collect(10000 : 1000 : 10000))

    # # infinitesimal inverse temperature step and inverse temperatures
    # δβ = 1e-4
    # inverseTemperatures = δβ .* vcat(collect(1 : 1 : 99), collect(100 : 10 : 990), collect(1000 : 100 : 9900), collect(10000 : 1000 : 10000))

    # initialization method (0 ==> iPESO(β = 0), 1 ==> iPESO(β = β_max))
    initMethod = 0

    # set tensorUpdateFlag
    tensorUpdateFlag = "iPESO"

    # flags to run simple update and mean-field expectation values
    computeFlags = [0, 1]


    #--------------------------------------------------------------------
    # sun simple update to get ground state wave function
    #--------------------------------------------------------------------

    # run iPEPS simple update
    if computeFlags[1] == 1

        # loop over all chiB
        for chiBIdx = eachindex(bondDimsB)

            # get chiB
            chiB = bondDimsB[chiBIdx]

            # loop over all magneticFields
            for idxHz = eachindex(hamCouplingsHz)

                # get magnetic field value
                Hz = hamCouplingsHz[idxHz]
                modelParameters = Vector{Float64}([J1, J2, J3, J4, J5, Hz])

                # construct modelDict
                modelDict = Dict(
                    "latticeName" => latticeName, 
                    "modelName" => modelName,
                    "setSym" => setSym,
                    "physicalSpin" => physicalSpin,
                    "Lx" => Lx,
                    "Ly" => Ly,
                    "unitCell" => unitCell,
                    "modelParameters" => modelParameters,
                    "invTemps" => inverseTemperatures,
                    "chiB" => chiB,
                    "dBeta" => δβ
                )
                println(modelDict)

                # run iPEPS simple update
                simpleUpdate_iPESO(modelDict, initMethod, verbosePrint = true)

            end

        end

    end


    #--------------------------------------------------------------------
    # compute mean-field expectation values
    #--------------------------------------------------------------------

    # loop over all chiB
    for chiBIdx = eachindex(bondDimsB)

        # get chiB
        chiB = bondDimsB[chiBIdx]

        # loop over all magneticFields
        for idxHz = eachindex(hamCouplingsHz)

            # get magnetic field value
            Hz = hamCouplingsHz[idxHz]
            modelParameters = Vector{Float64}([J1, J2, J3, J4, J5, Hz])

            # loop over all inverse temperatures
            for betaTIdx = eachindex(inverseTemperatures)
    
                # get β
                β = inverseTemperatures[betaTIdx]

                # construct modelDict
                modelDict = Dict(
                    "latticeName" => latticeName,
                    "modelName" => modelName,
                    "setSym" => setSym,
                    "physicalSpin" => physicalSpin,
                    "Lx" => Lx,
                    "Ly" => Ly,
                    "unitCell" => unitCell,
                    "modelParameters" => modelParameters,
                    "betaT" => β,
                    "chiB" => chiB,
                    "dBeta" => δβ,
                )
                # println(modelDict)

                # compute ground state energy and magnetization
                if computeFlags[2] == 1
                    computeGroundStateEnergy_MF(modelDict, tensorUpdateFlag)
                end

            end

        end

    end

end