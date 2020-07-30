#=
#
#   InstallRequiredPackages.jl
#
#   Install Julia packages required by the Identifiability module
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       e: ap.browning@qut.edu.a
#       w: alexbrowning.me
#
=#

using Pkg;

Pkg.add("Random")
Pkg.add("Distributions")
Pkg.add("KernelDensity")
Pkg.add("LinearAlgebra")
Pkg.add("PyPlot")
Pkg.add("StatsBase")
Pkg.add("Statistics")
Pkg.add("Interpolations")
Pkg.add("JLD2")
