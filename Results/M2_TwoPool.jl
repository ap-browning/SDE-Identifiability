#=

    M2_TwoPool.jl

    Perform practical identifiability analysis on the Two-Pool model using
    PM-MCMC

=#

using Identifiability
using LinearAlgebra
using PyPlot
using JLD2

include("IdentifiabilityProjectOptions.jl")

# Load results from .JLD2 file?
load = true

## DEFINE MODEL AND GENERATE SYNTHETIC DATA

    # Two pool model

    # θ₁: X₁ → ∅
    # θ₂: X₂ → ∅
    # θ₃: X₁ → X₂
    # θ₄: X₂ → X₁

    # Y = X + ξ, ξ ~ N(0,θ₅^2)

    # Propensities
    a  = (X,t,θ) -> [ θ[1]*X[1],
                      θ[2]*X[2],
                      θ[3]*X[1],
                      θ[4]*X[2] ];

    # Propensities (log rates)
    a2 = (X,t,θ) -> [ exp(θ[1])*X[1],
                      exp(θ[2])*X[2],
                      exp(θ[3])*X[1],
                      exp(θ[4])*X[2] ];

    # Stoichiometries
    ν  = [-1.0  0.0 -1.0  1.0;
           0.0 -1.0  1.0 -1.0];

    # Observation process
    g_rand = (Xt,t,θ) -> Xt[1] +  θ[5] * randn()
    g_lpdf = (Yt,Xt,t,θ) -> logpdf(Normal(Xt[1],θ[5]), Yt[1])

    # "True" parameters and initial condition
    θ  = [0.1,0.2,0.2,0.5,2.0]
    x0 = [100.0,0.0]

    # Simulate synthetic data (10 experiments, each with 10 observations)
    E    = 5
    X0   = θ -> x0 * ones(1,E)                      # initial conditions as a function of parameters
    T    = collect(range(1.0,10.0,step=1.0))        # observation times
    X,Y  = SimulateSSA(a,ν,X0,g_rand,θ,T,seed=20)   # simulate SSA

## PERFORM PM-MCMC

    # Prior
    PriorMin = [0.0,0.0,0.0,0.0, 0.0]
    PriorMax = [0.5,2.0,1.0,2.0,10.0]
    log_π    = GetLogPrior("independent-uniform",PriorMin,PriorMax)

    # Prior (logged)
    PriorMin2 = [-7.0,-7.0,-7.0,-7.0, 0.0]
    PriorMax2 = [ 2.0, 2.0, 2.0, 2.0,10.0]
    log_π2    = GetLogPrior("independent-uniform",PriorMin2,PriorMax2)

    # Pilot proposal (1/100th prior variance)
    Σpilot   = 1/1200 * diagm((PriorMax - PriorMin).^2)     # untransformed
    Σpilot2  = 1/1200 * diagm((PriorMax2 - PriorMin2).^2)   # logged

    # MCMC settings
    C   = 4                             # Number of chains
    S   = [10000,30000]                 # Number of MCMC iterations ([pilot chains, tuned chains])
    R   = 200                           # Number of particles
    θ0  = PriorSample("independent-uniform",C,PriorMin,PriorMax,seed=21)    # Initialise chain
    θ02 = PriorSample("independent-uniform",C,PriorMin2,PriorMax2,seed=22)  # Initialise chain (logged)
    Δt  = 0.02                          # Euler/Euler-Maruyama timestep
    Ti  = Int.(floor.(T / Δt))          # Timesteps in units of Δt

    # Simulate MCMC chains
    if load
        @load "Saved/M2_TwoPool.jld2" Θs_pilot Ls_pilot Θs2_pilot Ls2_pilot Σs Θs Ls
    else

        # SDE pilot chains (untransformed)
        @time Θs_pilot,Ls_pilot = PseudoMarginalMH(a,ν,X0,g_lpdf,Y,Ti,Δt,R,θ0,Σpilot,log_π,S[1],C)

        # SDE pilot chains (logged)
        @time Θs_pilot2,Ls_pilot2 = PseudoMarginalMH(a2,ν,X0,g_lpdf,Y,Ti,Δt,R,θ02,Σpilot2,log_π2,S[1],C)

        # Tune proposals
        Σs = OptimalProposal(Θs_pilot)

        # SDE tuned chains (30000 iterations)
        @time Θs,Ls = PseudoMarginalMH(a,ν,X0,g_lpdf,Y,Ti,Δt,R,θ0,Σs,log_π,S[2],C)

    end


## VISUALISE RESULTS

    ## Figure 5a: Untransformed pilot traces
    fig5a = figure("Figure5a",figsize=(3.4, 2.55)); clf()

    PlotTraces(Θs_pilot,Ls_pilot,fig=fig5a,truevals=θ,
        PMin=PriorMin,PMax=PriorMax,
        LLElim=[-200,-100])


    ## Figure 5b: Transformed p ilot traces
    fig5b = figure("Figure5b",figsize=(3.4, 2.55)); clf()

    PlotTraces(Θs2_pilot,Ls2_pilot,fig=fig5b,truevals=[log.(θ[1:4]);θ[5]],
        PMin=PriorMin2,PMax=PriorMax2,
        LLElim=[-200,-100])


    ## Figure 6: Untransformed, tuned, scatter plot matrix
    fig6 = figure("Figure6",figsize=(6.9, 5.2)); clf()

    options = Dict( :lims=>     PriorMax,
                    :truevals=> θ,
                    :skip=>     300,
                    :burnin=>   3000,
                    :s_alpha=>  0.2,
                    :π_pdf=>    x -> MarginalPriorPDF(x,"independent-uniform",PriorMin,PriorMax)
                  )

    PlotScatterMatrix(Θs,fig=fig6;options...)

    # Plot ACF function on top right (first chain)
    ax = subplot(5,3,3);
    ρ  = autocor(Θs[1,:,3002:end]',0:400)
    ax.plot(0:400,ρ)
    ax.plot([-20,420],[0.0,0.0],"k:")
    ax.set_xlim([-20,420]); ax.set_ylim([-0.15,1.05])
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    legend(["θ1","θ2","θ3","θ4","σerr"])

    display(fig6)

## Diagnostics and credible intervals

R̂s,Ss = Diagnostics(Θs,burnin=3000)
CIs   = CredibleIntervals(Θs,burnin=3000,skip=1)
