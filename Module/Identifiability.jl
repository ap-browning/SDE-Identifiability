#=
#
#   Identifiability.jl
#
#   Module to perform practical identifiability analysis on stochastic differential
#   equation (SDE models)
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       e: ap.browning@qut.edu.a
#       w: alexbrowning.me
#
=#


module Identifiability

    using Random
    using KernelDensity
    using Distributions
    using LinearAlgebra
    using PyPlot
    using StatsBase
    using Random
    using .Threads
    using Interpolations

    # Plots
    export PlotScatterMatrix
    export PlotTraces

    # MCMC
    export MetropolisHastings
    export DeterministicMH
    export PseudoMarginalMH

    export LogNormalParams
    export GetLogPrior
    export MarginalPriorPDF
    export PriorSample
    export OptimalProposal
    export Diagnostics
    export CredibleIntervals
    export PosteriorPredictiveQuantiles

    # Simulate models
    export SimulateSDE
    export SimulateSSA

    # Inference
    include("Inference/MetropolisHastings.jl")
    include("Inference/PMLogLikeParticleFilter.jl")
    include("Inference/Priors.jl")
    include("Inference/OptimalProposal.jl")
    include("Inference/Diagnostics.jl")
    include("Inference/CredibleIntervals.jl")

    # Models
    include("Models/Euler.jl")
    include("Models/EulerMaruyama.jl")
    include("Models/SimulateSDE.jl")
    include("Models/SimulateSSA.jl")
    include("Models/PosteriorPredictive.jl")

    # Plots
    include("Plots/PlotScatterMatrix.jl")
    include("Plots/PlotTraces.jl")

end

#=

    LIST OF COMMON VARIABLES IN THE MODULE

    symbol  :   meaning                                     : notes
    -------------------------------------------------------------------------------------
    a       :   propensity function                         : size(a(⋅))        = (N,)
    ν       :   stoichiometric matrix                       : size(ν)           = (N,Q)
    α       :   drift term                                  : size(α(⋅))        = (N,)
    σ       :   diffusion term                              : size(σ(⋅))        = (N,Q)
    g_rand  :   sample from observation function            : size(g_rand(⋅))   = (N,)
    g_pdf   :   observation probability density function    : size(g_pdf(⋅))    = ()
    θ       :   parameter vector                            : size(θ)           = (D,)
    T       :   vector of observation times                 : size(T)           = (O,)
    Ti      :   vector of observation times (units of Δt)   : size(Ti)          = (O,)
    Δt      :   Euler and Euler-Maruyama timestep
    -------------------------------------------------------------------------------------
    X       :   state at all times, all experiments         : size(X)           = (N,O,E)
    Xt      :   state at single time                        : size(Xt)          = (N,)
    Y       :   observables at all times, all experiments   : size(Y)           = (M,O,E)
    Yt      :   observables at single time                  : size(Yt)          = (M,)
    -------------------------------------------------------------------------------------
    N       :   number of state equations
    M       :   number of observables
    Q       :   number of reactions
    E       :   number of experiments
    O       :   number of observation points
    R       :   number of replicates
    -------------------------------------------------------------------------------------
    Σ       :   proposal covariance matrix (MV normal)      : size(Σ)           = (D,D)
    C       :   number of MCMC chains
    θ0      :   initial chain location                      : size(θ0)          = (D,)
    S       :   length of each chain (excl. first)
    Θ       :   MCMC samples (incl. first)                  : size(Θ)           = (C,D,S+1)
    L       :   Log-likelihood estimate at each sample      : size(L)           = (C,S+1)
    log_π   :   Log-prior distribution

=#
