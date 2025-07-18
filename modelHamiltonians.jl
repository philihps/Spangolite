#!/usr/bin/env julia

include("utilities.jl")

#-------------------------------------------------------------------------------------

function getHamiltonianGates_SP_iPESS_6Site_SU0(physicalSpin::Float64, modelParameters::Vector{Float64})

    # get Pauli matrices for physical spin S
    Sx, Sy, Sz, Sm, Sp, Id = getSpinOperators(physicalSpin)

    # construct magnetic fields
    Z1 = Sz ⊗ Id ⊗ Id ⊗ Id ⊗ Id ⊗ Id
    Z2 = Id ⊗ Sz ⊗ Id ⊗ Id ⊗ Id ⊗ Id
    Z3 = Id ⊗ Id ⊗ Sz ⊗ Id ⊗ Id ⊗ Id
    Z4 = Id ⊗ Id ⊗ Id ⊗ Sz ⊗ Id ⊗ Id
    Z5 = Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sz ⊗ Id
    Z6 = Id ⊗ Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sz

    # dissect modelParameters
    J1 = modelParameters[1]
    J2 = modelParameters[2]
    J3 = modelParameters[3]
    J4 = modelParameters[4]
    J5 = modelParameters[5]
    Hz = modelParameters[6]

    # construct necessary Hamiltonian terms
    H12 = Sx ⊗ Sx ⊗ Id ⊗ Id ⊗ Id ⊗ Id + Sy ⊗ Sy ⊗ Id ⊗ Id ⊗ Id ⊗ Id + Sz ⊗ Sz ⊗ Id ⊗ Id ⊗ Id ⊗ Id
    H13 = Sx ⊗ Id ⊗ Sx ⊗ Id ⊗ Id ⊗ Id + Sy ⊗ Id ⊗ Sy ⊗ Id ⊗ Id ⊗ Id + Sz ⊗ Id ⊗ Sz ⊗ Id ⊗ Id ⊗ Id
    H14 = Sx ⊗ Id ⊗ Id ⊗ Sx ⊗ Id ⊗ Id + Sy ⊗ Id ⊗ Id ⊗ Sy ⊗ Id ⊗ Id + Sz ⊗ Id ⊗ Id ⊗ Sz ⊗ Id ⊗ Id
    H15 = Sx ⊗ Id ⊗ Id ⊗ Id ⊗ Sx ⊗ Id + Sy ⊗ Id ⊗ Id ⊗ Id ⊗ Sy ⊗ Id + Sz ⊗ Id ⊗ Id ⊗ Id ⊗ Sz ⊗ Id
    H16 = Sx ⊗ Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sx + Sy ⊗ Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sy + Sz ⊗ Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sz
    H23 = Id ⊗ Sx ⊗ Sx ⊗ Id ⊗ Id ⊗ Id + Id ⊗ Sy ⊗ Sy ⊗ Id ⊗ Id ⊗ Id + Id ⊗ Sz ⊗ Sz ⊗ Id ⊗ Id ⊗ Id
    H24 = Id ⊗ Sx ⊗ Id ⊗ Sx ⊗ Id ⊗ Id + Id ⊗ Sy ⊗ Id ⊗ Sy ⊗ Id ⊗ Id + Id ⊗ Sz ⊗ Id ⊗ Sz ⊗ Id ⊗ Id
    H25 = Id ⊗ Sx ⊗ Id ⊗ Id ⊗ Sx ⊗ Id + Id ⊗ Sy ⊗ Id ⊗ Id ⊗ Sy ⊗ Id + Id ⊗ Sz ⊗ Id ⊗ Id ⊗ Sz ⊗ Id
    H26 = Id ⊗ Sx ⊗ Id ⊗ Id ⊗ Id ⊗ Sx + Id ⊗ Sy ⊗ Id ⊗ Id ⊗ Id ⊗ Sy + Id ⊗ Sz ⊗ Id ⊗ Id ⊗ Id ⊗ Sz
    H34 = Id ⊗ Id ⊗ Sx ⊗ Sx ⊗ Id ⊗ Id + Id ⊗ Id ⊗ Sy ⊗ Sy ⊗ Id ⊗ Id + Id ⊗ Id ⊗ Sz ⊗ Sz ⊗ Id ⊗ Id
    H35 = Id ⊗ Id ⊗ Sx ⊗ Id ⊗ Sx ⊗ Id + Id ⊗ Id ⊗ Sy ⊗ Id ⊗ Sy ⊗ Id + Id ⊗ Id ⊗ Sz ⊗ Id ⊗ Sz ⊗ Id
    H36 = Id ⊗ Id ⊗ Sx ⊗ Id ⊗ Id ⊗ Sx + Id ⊗ Id ⊗ Sy ⊗ Id ⊗ Id ⊗ Sy + Id ⊗ Id ⊗ Sz ⊗ Id ⊗ Id ⊗ Sz
    H45 = Id ⊗ Id ⊗ Id ⊗ Sx ⊗ Sx ⊗ Id + Id ⊗ Id ⊗ Id ⊗ Sy ⊗ Sy ⊗ Id + Id ⊗ Id ⊗ Id ⊗ Sz ⊗ Sz ⊗ Id
    H46 = Id ⊗ Id ⊗ Id ⊗ Sx ⊗ Id ⊗ Sx + Id ⊗ Id ⊗ Id ⊗ Sy ⊗ Id ⊗ Sy + Id ⊗ Id ⊗ Id ⊗ Sz ⊗ Id ⊗ Sz
    H56 = Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sx ⊗ Sx + Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sy ⊗ Sy + Id ⊗ Id ⊗ Id ⊗ Id ⊗ Sz ⊗ Sz

    # set conversion factor between meV (milli-electron-volt) and K (Kelvin)
    g = 2.00231930436
    μB = 0.01 * 5.7883818012 # in (meV/T)
    kB = 8.617333262 * 1e-2 # in (meV/K)
    gateH_△ = + J1/2 * (H12 + H34 + H56) + J2 * (H15 + H23 + H46) + J4 * (H24 + H25 + H45) - 0.5 * Hz * g * μB * (Z1 + Z2 + Z3 + Z4 + Z5 + Z6) / kB
    gateH_▽ = + J1/2 * (H12 + H34 + H56) + J3 * (H13 + H26 + H45) + J5 * (H23 + H25 + H35) - 0.5 * Hz * g * μB * (Z1 + Z2 + Z3 + Z4 + Z5 + Z6) / kB

    # fuse dimer sites
    physVecSpace = ComplexSpace(Int(2 * physicalSpin + 1))
    isoD = TensorKit.isomorphism(fuse(physVecSpace, physVecSpace), physVecSpace ⊗ physVecSpace)
    isoU = TensorKit.isomorphism(physVecSpace ⊗ physVecSpace, fuse(physVecSpace, physVecSpace))
    @tensor gateH_△[-1 -2 -3; -4 -5 -6] := isoD[-1, 1, 2] * isoD[-2, 3, 4] * isoD[-3, 5, 6] * gateH_△[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * isoU[7, 8, -4] * isoU[9, 10, -5] * isoU[11, 12, -6]
    @tensor gateH_▽[-1 -2 -3; -4 -5 -6] := isoD[-1, 1, 2] * isoD[-2, 3, 4] * isoD[-3, 5, 6] * gateH_▽[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * isoU[7, 8, -4] * isoU[9, 10, -5] * isoU[11, 12, -6]
    return gateH_△, gateH_▽

    # funtion return
    return real(gateH_△), real(gateH_▽)

end
function getSuzukiTrotterGate_SP_iPESS_6Site_SU0(physicalSpin::Float64, modelParameters::Vector{Float64}, dt::Float64)

    # get Hamiltonian gate and apply Suzuki Trotter step
    gateH_△, gateH_▽ = getHamiltonianGates_SP_iPESS_6Site_SU0(physicalSpin, modelParameters)
    gateH_△ = exp(-dt * gateH_△)
    gateH_▽ = exp(-dt * gateH_▽)
    return [gateH_△, gateH_▽]

end

#-------------------------------------------------------------------------------------