#=

    M1_BirthDeath.jl

    Perform practical identifiability analysis on the Birth-Death model using
    PM-MCMC

=#

using Identifiability
using LinearAlgebra
using PyPlot
using JLD2
using KernelDensity

include("IdentifiabilityProjectOptions.jl")

# Load results from .JLD2 file?
load = true

## DEFINE MODEL AND GENERATE SYNTHETIC DATA

    # Birth death model

    # θ₁: X → 2X
    # θ₂: X → ∅

    # Y = ξ X, ξ ~ N(1,θ[3]^2)

    # Propensities
    a  = (X,t,θ) -> [θ[1]*X[1],
                     θ[2]*X[1]]

    # Stoichiometries
    ν  = [1.0 -1.0]

    # Observation process
    g_rand = (Xt,t,θ)       -> [Xt[1] * (1 + θ[3] * randn())]
    g_lpdf = (Yt,Xt,t,θ)    -> logpdf(Normal(Xt[1],Xt[1] * θ[3]), Yt[1])

    # "True" parameters and initial condition
    θ  = [0.2,0.1,0.05]
    x0 = 50.0

    # Simulate synthetic data (10 experiments, each with 10 observations)
    E    = 10
    X0   = θ -> x0 * ones(1,E)                      # initial conditions as a function of parameters
    T    = collect(range(1.0,10.0,step=1.0))        # observation times
    X,Y  = SimulateSSA(a,ν,X0,g_rand,θ,T,seed=10)   # simulate SSA

## PERFORM PM-MCMC

    # Prior
    PriorMin = [0.0,0.0,0.0]
    PriorMax = [0.6,0.6,0.3]
    log_π    = GetLogPrior("independent-uniform",PriorMin,PriorMax)

    # Pilot proposal (1/100th prior variance)
    Σpilot   = 1/1200 * diagm((PriorMax - PriorMin).^2)

    # MCMC settings
    C   = 4                         # Number of chains
    S   = 10000                     # Number of MCMC iterations
    R   = 200                       # Number of particles
    θ0  = PriorSample("independent-uniform",C,PriorMin,PriorMax,seed=11)    # Initialise chain
    Δt  = 0.01                      # Euler/Euler-Maruyama timestep
    Ti  = Int.(floor.(T / Δt))      # Timesteps in units of Δt

    # Simulate MCMC chains
    if load
        @load "Saved/M1_BirthDeath.jld2" runtime Θo_pilot Lo_pilot Θs_pilot Ls_pilot Σo Σs Θo Lo Θs Ls
    else

        runtime = @elapsed begin

            # ODE pilot chains
            @time Θo_pilot,Lo_pilot = DeterministicMH(a,ν,X0,g_lpdf,Y,Ti,Δt,θ0,Σpilot,log_π,S,C,seed=12)

            # SDE pilot chains
            @time Θs_pilot,Ls_pilot = PseudoMarginalMH(a,ν,X0,g_lpdf,Y,Ti,Δt,R,θ0,Σpilot,log_π,S,C,seed=13)

            # Tune proposals
            Σo = OptimalProposal(Θo_pilot)
            Σs = OptimalProposal(Θs_pilot)

            # ODE tuned chains
            @time Θo,Lo = DeterministicMH(a,ν,X0,g_lpdf,Y,Ti,Δt,θ0,Σo,log_π,S,C,seed=14)

            # SDE tuned chains
            @time Θs,Ls = PseudoMarginalMH(a,ν,X0,g_lpdf,Y,Ti,Δt,R,θ0,Σs,log_π,S,C,seed=15)

        end

    end


## VISUALISE RESULTS

    # Figure 4a and 4b side-by-side. 4a: ODE model, 4b: SDE model

    # Scatter plot matrix options
    options = Dict( :lims=>     PriorMax,
                    :truevals=> θ,
                    :skip=>     100,
                    :burnin=>   3000,
                    :s_alpha=>  0.2,
                    :π_pdf=>    x -> MarginalPriorPDF(x,"independent-uniform",PriorMin,PriorMax)
                  )

    # Figure 4: MCMC results for ODE model
    fig4a = figure("Figure4a",figsize=(3.4, 2.55)); clf()
    PlotScatterMatrix(Θo,fig=fig4a;options...)

    # Figure 5: MCMC results for SDE model
    fig4b = figure("Figure4b",figsize=(3.4, 2.55)); clf()
    PlotScatterMatrix(Θs,fig=fig4b;options...)


## STATISTICS AND CREDIBLE INTERVALS

    R̂o,So = Diagnostics(Θo,burnin=3000)
    R̂s,Ss = Diagnostics(Θs,burnin=3000)
    CIo   = CredibleIntervals(Θo,burnin=3000,skip=1)
    CIs   = CredibleIntervals(Θs,burnin=3000,skip=1)
