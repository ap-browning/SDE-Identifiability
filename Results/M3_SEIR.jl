#=

    M3_SEIR.jl

    Perform practical identifiability analysis on the Epidemic model using
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

    # Epidemic model

    # X = [S,E,I,R]

    # θ₁I: S → E
    # θ₂:  E → I
    # θ₃:  I → R

    # Y₁ = ξ⋅θ₄⋅I, ξ ~ N(1,θ₆²)
    # Y₂ = ξ⋅θ₄⋅R, ξ ~ N(1,θ₆²)

    # Propensities
    a  = (X,t,θ) -> [ θ[1]*X[1]*X[3],
                      θ[2]*X[2],
                      θ[3]*X[3]];

    # Stoichiometries
    ν  = [-1.0  0.0  0.0;
           1.0 -1.0  0.0;
           0.0  1.0 -1.0;
           0.0  0.0  1.0];

    # Observation process, Y ~ g(Y | X,t,θ)
    g_rand = (Xt,t,θ) -> [ Xt[3] * (θ[5] + θ[6] * randn()),                      # Sample function
                           Xt[4] * (θ[5] + θ[6] * randn()) ]
    g_lpdf = (Yt,Xt,t,θ) -> logpdf(Normal(Xt[3] * θ[5], Xt[3] * θ[6]),Yt[1]) +   # Log PDF
                            logpdf(Normal(Xt[4] * θ[5], Xt[4] * θ[6]),Yt[2])

    # "True" parameters
    θ  = [0.01,0.15,0.1,10.0,0.5,0.05]

    # Simulate synthetic data (1 experiment, short data out to time 10, long to time 30)
    E     = 1
    X0    = θ -> [500 - θ[4],θ[4],10 / θ[5],10 / θ[5]] * ones(1,E)   # initial conditions as a function of parameters
    Tl    = collect(range(1.0,30.0,step=1.0))            # observation times
    Xl,Yl = SimulateSSA(a,ν,X0,g_rand,θ,Tl,seed=30)       # simulate SSA
    Ts    = Tl[1:10]        # Short data is first 10 days from long data
    Ys    = Yl[:,1:10,:]    #   ""

    # Extra short-time synthetic data where the number of exposed is monitored
    Xs    = Xl[:,1:10,:]
    Ys2   = zeros(3,10,1)
    Ys2[1:2,:,:] = Ys
    Ys2[3,:,:]   = Xs[2,:] .* (θ[5] .+ θ[6] * randn(10))
    g_lpdf2 = (Yt,Xt,t,θ) -> logpdf(Normal(Xt[3] * θ[5], Xt[3] * θ[6]),Yt[1]) +   # Log PDF
                             logpdf(Normal(Xt[4] * θ[5], Xt[4] * θ[6]),Yt[2]) +
                             logpdf(Normal(Xt[2] * θ[5], Xt[2] * θ[6]),Yt[3])

## PERFORM PM-MCMC

    # Prior
    PriorMin = [0.0,0.0,0.0,0.0, 0.2,0.0]
    PriorMax = [0.1,1.0,0.5,20.0,1.0,0.2]
    log_π    = GetLogPrior("independent-uniform",PriorMin,PriorMax)

    # Pilot proposal (1/100th prior variance)
    Σpilot   = 1/1200 * diagm((PriorMax - PriorMin).^2)     # untransformed

    # MCMC settings
    C   = 4                         # Number of chains
    S   = 10000                     # Number of MCMC iterations
    R   = 400                       # Number of particles
    θ0  = PriorSample("independent-uniform",C,PriorMin,PriorMax,seed=31)  # Initialise chain (Short data)
    Δt  = 0.02                      # Euler/Euler-Maruyama timestep
    Tis = Int.(floor.(Ts / Δt))
    Til = Int.(floor.(Tl / Δt))

    # Simulate MCMC chains
    if load
        @load "Saved/M3_SEIR.jld2" Θs_pilot Ls_pilot Θs2_pilot Ls2_pilot Θl_pilot Ll_pilot
    else

        # SDE short pilot chains
        @time Θs_pilot,Ls_pilot = PseudoMarginalMH(a,ν,X0,g_lpdf,Ys,Tis,Δt,R,θ0,Σpilot,log_π,S,C)

        # SDE short pilot chains (also monitor exposed)
        @time Θs2_pilot,Ls2_pilot = PseudoMarginalMH(a,ν,X0,g_lpdf2,Ys2,Tis,Δt,R,θ0,Σpilot,log_π,S,C)

        # SDE long pilot chains
        @time Θl_pilot,Ll_pilot = PseudoMarginalMH(a,ν,X0,g_lpdf,Yl,Til,Δt,R,θ0,Σpilot,log_π,S,C)

    end


## VISUALISE RESULTS

    ## Figure 7: View all pilot traces
     #  Column 1: Short-time data
     #  Column 2: Long-time data
     #  Column 3: Short-time data, view "full" state
    fig7,axs = subplots(7,3,figsize=(6.9, 5.2));

    Θ_pilot = [Θs_pilot,Θl_pilot,Θs2_pilot]
    L_pilot = [Ls_pilot,Ll_pilot,Ls2_pilot]
    Labels  = ["θ₁","θ₂","θ₃","E₀","μₒ","σₑ"]
    Titles  = ["Short-time","Long-time","Short-time + E(t)"]
    for i = 1:3

        Θ = Θ_pilot[i]
        L = L_pilot[i]

        for c = 1:4

            # Plot likelihood
            ax = axs[1,i]

            ax.plot((0:10000)/1e3,L[c,:],
                alpha   = (c == 1 ? 1.0 : 0.3),
                zorder  = (c == 1 ? 10 : 0))
            ax.set_xticklabels([])
            ax.set_xticklabels([])
            ax.set_ylim([-500,0])
            if i != 1
                ax.set_yticklabels([])
            else
                ax.set_ylabel("L̂")
            end
            ax.set_title(Titles[i])

            # Plot variables
            for n = 1:6

                ax = axs[n+1,i]

                ax.plot((0:10000)/1e3,Θ[c,n,:],
                    alpha   = (c == 1 ? 1.0 : 0.3),
                    zorder  = (c == 1 ? 10 : 1))

                ax.set_ylim([PriorMin[n],PriorMax[n]])

                if n != 6
                    ax.set_xticklabels([])
                else
                    ax.set_xlabel("Iteration ('000)")
                end

                if i != 1
                    ax.set_yticklabels([])
                else
                    ax.set_ylabel(Labels[n])
                end

                if c == 4
                    xlim = ax.get_xlim()
                    ax.plot(xlim,θ[n]*[1,1],"k:",zorder=20)
                    ax.set_xlim(xlim)
                end

                ax.set_yticks([PriorMin[n],(PriorMin[n]+PriorMax[n])/2,PriorMax[n]])

            end

        end

    end

    display(fig7)

    ## Figure 8: Posterior predictive check (this takes some time to run)
     #  Column 1: Short-time data
     #  Column 2: Long-time data
     #  Column 3: Short-time data, view "full" state
    fig8,axs = subplots(1,3,figsize=(6.9, 1.6));

    Tpred   = collect(range(0.0,32,length=301))
    reps    = 10000

    for i = 1:3

        ax = axs[i]

        # Compute quantiles
        Qs = PosteriorPredictiveQuantiles(Θ_pilot[i],[0.025,0.25,0.75,0.975],a,ν,X0,g_rand,Tpred,Δt,reps)

        # Plot quantiles
        CI95_Y2 = ax.fill_between(Tpred,Qs[2,:,1],Qs[2,:,4],color="#333333",alpha=0.3)
        CI50_Y2 = ax.fill_between(Tpred,Qs[2,:,2],Qs[2,:,3],color="#333333",alpha=0.3)
        CI95_Y1 = ax.fill_between(Tpred,Qs[1,:,1],Qs[1,:,4],color=pcols[3],alpha=0.3)
        CI50_Y1 = ax.fill_between(Tpred,Qs[1,:,2],Qs[1,:,3],color=pcols[3],alpha=0.3)

        Y1p, = ax.plot(vcat(0.0,Tl),vcat(10.0,Yl[1,:,1]),"-s",color="#037e28",markersize=3,linewidth=0.5)
        Y2p, = ax.plot(vcat(0.0,Tl),vcat(10.0,Yl[2,:,1]),"-^",color="#333333",markersize=3,linewidth=0.5)

        ax.set_ylim([0,400])
        ax.set_xlim([-1.5,31.5])

        if i == 1
            ax.legend(
                [CI95_Y1,   CI50_Y1,    CI95_Y2,    CI50_Y2,    Y1p,    Y2p],
                ["95% CI",  "50% CI",   "95% CI",   "50% CI",   "Y1",   "Y2"])
        else
            ax.set_yticklabels([])
        end

        ax.set_xlabel("t")
        ax.set_title(Titles[i])

    end

    display(fig8)
