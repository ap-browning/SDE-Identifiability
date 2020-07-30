#=

    M4_BetaIG.jl

    Perform practical identifiability analysis on the β-insulin-glucose model
    using PM-MCMC

=#

using Identifiability
using LinearAlgebra
using PyPlot
using JLD2

include("IdentifiabilityProjectOptions.jl")

# Load results from .JLD2 file?
load = true

fig5 = figure("Figure5",figsize=(3.4, 2.55)); clf()
fig7 = figure("Figure7",figsize=(3.4, 2.55)); clf()


## DEFINE MODEL AND GENERATE SYNTHETIC DATA

    # β-cell, insulin, glucose

    # X = [β,I,G]

    # Y₁ = β + ξ₁, ξ₁ ~ N(0,θ₃^2)
    # Y₂ = G + ξ₂, ξ₂ ~ N(0,θ₃^2)

    # Known parameters and functions
    μp = 0.021 / (24 * 60)
    μm = 0.025 / (24 * 60)
    η  = 7.85
    γ  = 0.3
    u0 = 1/30
    c  = 1e-3

    λp = G -> μp / (1.0 + (8.6 / G)^1.7)
    λm = G -> μm / (1.0 + (G / 4.8)^8.5)
    ρ  = G -> G^2.0 / (η^2.0 + G^2.0)
    u  = t -> t < 50.0 ? 0.2 : 0

    # Propensities
    a  = (X,t,θ) -> [ X[1] * λp(X[3]),      # ∅ → β
                      X[2] * λm(X[3]),      # β → ∅
                      θ[1] * X[1]*ρ(X[3]),  # ∅ → I
                      γ * X[2],             # I → ∅
                      u0,                   # ∅ → G
                      u(t),                 # ∅ → G
                      c * X[3],             # G → ∅
                      θ[2] * X[2] * X[3] ]  # G → ∅

    # Propensities (transformed)
    a2  = (X,t,θ) -> [ X[1] * λp(abs(X[3])),
                       X[2] * λm(abs(X[3])),
                       sqrt(θ[1]*θ[2]) * X[1]*ρ(abs(X[3])),
                       γ * X[2],
                       u0,
                       u(t),
                       c * X[3],
                       sqrt(θ[1]/θ[2]) * X[2] * X[3] ]

    # Stoichiometries
    ν  = [ 1.0  0.0  0.0
          -1.0  0.0  0.0
           0.0  1.0  0.0
           0.0 -1.0  0.0
           0.0  0.0  1.0
           0.0  0.0  1.0
           0.0  0.0 -1.0
           0.0  0.0 -1.0 ]
    ν  = copy(ν')

    # Noise scaling
    Nscale = [1.0,20.0,20.0]

    # State equations (scale noise by 1 / √N)
    α = (X,t,θ) -> ν*a(X,t,θ)
    σ = (X,t,θ) -> ν*diagm(a(X,t,θ).^(1/2)) ./ sqrt.(Nscale)

    # Transformed state equations
    α2 = (X,t,θ) -> ν*a2(X,t,θ)
    σ2 = (X,t,θ) -> ν*diagm(a2(X,t,θ).^(1/2)) ./ sqrt.(Nscale)

    # Observation process
    g_rand = (Xt,t,θ) -> [Xt[1] + θ[3] * randn(),
                          Xt[3] + θ[3] * randn()]
    g_lpdf = (Yt,Xt,t,θ) -> logpdf(Normal(Xt[1],θ[3]),Yt[1]) +
                            logpdf(Normal(Xt[3],θ[3]),Yt[2])


    # "True" parameters (untransformed) and initial condition
    θ  = [3e-2,5e-4,1.0]
    x0 = [322.0,10.0,5.0]

    # "True" parameters (transformed)
    θ̃  = [θ[1]*θ[2],θ[1]/θ[2],θ[3]]

    # Numerical parameters
    Δt = 0.2

    # Simulate synthetic data (5 experiments, each with 15 observations)
    E     = 5
    X0    = θ -> x0 * ones(1,E)                         # initial conditions as a function of parameters
    T     = collect(range(10.0,150.0,step=10.0))        # observation times
    X,Y,  = SimulateSDE(α,σ,X0,g_rand,θ,T,Δt;seed=40)   # simulate SDE


## PERFORM PM-MCMC

    # Prior
    PriorMin = [0.0, 0.0, 0.0]
    PriorMax = [1e-1,2e-3,5.0]
    log_π    = GetLogPrior("independent-uniform",PriorMin,PriorMax)

    # Prior (transformed)
    PriorMin2 = [0.0, 0.0,  0.0]
    PriorMax2 = [3e-5,100.0,5.0]
    log_π2    = GetLogPrior("independent-uniform",PriorMin2,PriorMax2)

    # Pilot proposal (1/100th prior variance)
    Σpilot   = 1/1200 * diagm((PriorMax - PriorMin).^2)     # untransformed
    Σpilot2  = 1/1200 * diagm((PriorMax2 - PriorMin2).^2)   # logged

    # MCMC settings
    C   = 4                         # Number of chains
    S   = [100_000,10_000]          # Number of MCMC iterations
    R   = 200                       # Number of particles
    θ0  = PriorSample("independent-uniform",C,PriorMin,PriorMax,seed=41)    # Initialise chain
    θ02 = PriorSample("independent-uniform",C,PriorMin2,PriorMax2,seed=42)  # Initialise chain (logged)
    Δt  = 0.2                       # Euler/Euler-Maruyama timestep
    Ti  = Int.(floor.(T / Δt))      # Timesteps in units of Δt

    # Simulate MCMC chains
    if load
        @load "Saved/M4_BetaIG.jld2" Θo_pilot Lo_pilot Θo_pilot2 Lo_pilot2 Θs2_pilot Ls2_pilot Σs2 Θs2 Ls2
    else

        # ODE pilot chains (untransformed)
        @time Θo_pilot,Lo_pilot = DeterministicMH(a,ν,X0,g_lpdf,Y,Ti,Δt,θ0,Σpilot,log_π,S[1],C,seed=43)

        # ODE pilot chains (transformed)
        @time Θo_pilot2,Lo_pilot2 = DeterministicMH(a2,ν,X0,g_lpdf,Y,Ti,Δt,θ02,Σpilot2,log_π2,S[1],C,seed=44)

        # SDE pilot chains
        @time Θs2_pilot,Ls2_pilot = PseudoMarginalMH(α2,σ2,X0,g_lpdf,Y,Ti,Δt,R,θ02,Σpilot2,log_π2,S[2],C,seed=45)

        # Tune proposals
        Σs2 = OptimalProposal(Θs2_pilot)

        # SDE tuned chains
        @time Θs2,Ls2 = PseudoMarginalMH(α2,σ2,X0,g_lpdf,Y,Ti,Δt,R,θ02,Σs2,log_π2,S[2],C,seed=46)

    end


## VISUALISE RESULTS

    # Figure 9, ODE posterior figure
    fig9 = figure("Figure11",figsize=(3.4, 2.55)); clf()

        ## Fig 9a: Untransformed

        ### Construct KDE
        Θ = Θo_pilot[:,:,3002:end]  # Discard first 3000 samples after first
        samples = zeros(2,C*(S[1] - 3000))
        for i = 1:2
            samples[i,:] = Θ[:,i,:][:]
        end
        B = kde(samples')

        ### Calculate kernel density
        p1 = range(PriorMin[1],PriorMax[1],length=200)
        p2 = range(PriorMin[2],PriorMax[2],length=200)
        f  = pdf(B,p1,p2)

        ### Plot contour
        ax = subplot(1,2,1)
        c1 = ax.contourf(p1,p2,f',extend="min")
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))
        ax.set_xlabel("θ1")
        ax.set_ylabel("θ2")
        ax.set_yticks(0:0.001:0.002)
        ax.scatter(θ[1],θ[2],marker="v",color="w")

        ## Fig 9b: Transformed

        ### Construct KDE
        Θ = Θo_pilot2[:,:,3002:end]  # Discard first 3000 samples after first
        samples = zeros(2,C*(S[1] - 3000))
        for i = 1:2
            samples[i,:] = Θ[:,i,:][:]
        end
        B = kde(samples')

        ### Calculate kernel density
        p1 = range(PriorMin2[1],PriorMax2[1],length=200)
        p2 = range(PriorMin2[2],PriorMax2[2],length=200)
        f  = pdf(B,p1,p2)

        ### Plot contour
        ax = subplot(1,2,2)
        c2 = ax.contourf(p1/1e-5,p2,f',extend="min")
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))
        ax.set_xlabel("θ̃1/1e-5")
        ax.set_ylabel("θ̃2")
        ax.set_yticks(0:25:100)
        ax.set_xticks((0:0.00001:0.00003)./1e-5)
        ax.scatter(θ̃[1]/1e-5,θ̃[2],marker="v",color="w")

        # Show figure 11
        plt.tight_layout()
        display(fig9)


    # Figure 10, SDE pilot traces
    fig10 = figure("Figure12",figsize=(3.4, 3.0)); clf()

        PlotTraces(Θs2_pilot,Ls2_pilot,fig=fig10,truevals=θ̃,
            PMin=PriorMin2,PMax=PriorMax2,singlecol=true,
            LLElim=[-400,-100],LLEticks=-400.0:100.0:-100)

        # Show figure 12
        display(fig10)


    # Figure 11, SDE tuned scatter plot matrix
    fig11 = figure("Figure13",figsize=(6.9, 5.2)); clf()

        options = Dict( :lims=>     PriorMax2,
                        :truevals=> θ̃,
                        :skip=>     100,
                        :burnin=>   3000,
                        :s_alpha=>  0.2,
                        :π_pdf=>    x -> MarginalPriorPDF(x,"independent-uniform",PriorMin2,PriorMax2)
                  )

        PlotScatterMatrix(Θs2,fig=fig11;options...)

        # Plot ACF function on top right (first chain)
        ax = subplot(3,2,2);
        ρ  = autocor(Θs2[1,:,3002:end]',0:100)
        ax.plot(0:100,ρ)
        ax.plot([-10,110],[0.0,0.0],"k:")
        ax.set_xlim([-10,110]); ax.set_ylim([-0.15,1.05])
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        legend(["θ1","θ2","σerr"])

        # Show figure 13
        display(fig11)


## CREDIBLE INTERVALS

    CIo2 = CredibleIntervals(Θs2_pilot,burnin=3000,skip=1)
