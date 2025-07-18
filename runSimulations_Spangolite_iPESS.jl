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
include("simpleUpdate_iPESS.jl")
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

    # set bulk bond dimension
    bondDimsB = collect(1 : 1)
    
    # set simple update and CTMRG convergence
    convTolB = 1e-6
    convTolE = 1e-6
    
    # set tensorUpdateFlag
    tensorUpdateFlag = "iPESS"

    # initialize tensors (0 --> random, 1 --> smaller chiB)
    initMethod = 0

    # flags to run simple update, CTMRG, full-environment and mean-field expectation values 
    computeFlags = [1, 1, 1, 1]

    # set main directory path
    mainDirPath = @sprintf("numFiles_%s/%s/%s/S_%0.1f/%s/Lx_%d_Ly_%d", tensorUpdateFlag, latticeName, modelName, physicalSpin, setSym, Lx, Ly)

    # loop over all chiB
    for chiBIdx = eachindex(bondDimsB)

        # get chiB and set chiE
        chiB = bondDimsB[chiBIdx]
        if chiB == 1
            chiE = 1
        elseif 1 < chiB <= 8
            chiE = chiB * (chiB - 0)
        else
            chiE = chiB * (chiB - 2)
        end

        # loop over all magnetic fields
        for idxHz = eachindex(hamCouplingsHz)

            # get magnetic field value
            Hz = hamCouplingsHz[idxHz]
            modelParameters = Vector{Float64}([J1, J2, J3, J4, J5, Hz])

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
                "convTolB" => convTolB, 
                "convTolE" => convTolE, 
            )
            println(modelDict)

            # run iPEPS simple update
            if computeFlags[1] == 1
                simpleUpdate_iPESS(modelDict, initMethod, verbosePrint = true)
            end

            # set simpleUpdate directory path
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

            # run CTMRG routine
            if computeFlags[2] == 1
                isfile(fileStringSimpleUpdate) && computeEnvironment(modelDict, tensorUpdateFlag)
            end

            # compute expectation values
            if computeFlags[3] == 1
                isfile(fileStringSimpleUpdate) && computeExpectationValues(modelDict, tensorUpdateFlag)
            end

            # compute mean-field expectation values
            if computeFlags[4] == 1
                isfile(fileStringSimpleUpdate) && computeGroundStateEnergy_MF(modelDict, tensorUpdateFlag)
            end

        end

    end

end