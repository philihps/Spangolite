# Spangolite

This repository contains code that can be used to reproduce the results from the paper "Tensor network analysis of the maple-leaf antiferromagnet spangolite" by Philipp Schmoll, Harald O. Jeschke and Yasir Iqbal. The paper is currently available as a preprint [2404.14905](https://arxiv.org/abs/2404.14905).

The code is based on the [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) package and provides the following simulation algorithms for the maple-leaf lattice:
- simple update on infinite projected entangled simplex states to simulate ground state properties
- simple update on infinite projected entangled simplex operators to simulate thermal state properties
- CTMRG code to compute effective environments for infinite PEPS (ground states only)
- modified CTMRG code to compute structure factors for infinite PEPS (ground states only)
