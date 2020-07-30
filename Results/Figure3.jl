#=

    Figure3.jl

    Produce 100 realisations of the state equations, and generate synthetic data
    for each of the models considered

=#

using Identifiability
using LinearAlgebra
using PyPlot

include("IdentifiabilityProjectOptions.jl")

fig3 = figure("Data",figsize=(6.7,2.75)); clf()


## MODEL 1

    # Birth death model

    # θ₁: X → 2X
    # θ₂: X → ∅

    # Y = ξ X, ξ ~ N(1,θ₃^2)

    # Propensities
    a  = (X,t,θ) -> [θ[1]*X[1],
                     θ[2]*X[1]]

    # Stoichiometries
    ν  = [1.0 -1.0]

    # Observation process
    g_rand = (Xt,t,θ) -> Xt * (1 + θ[3] * randn())

    # "True" parameters and initial condition
    θ  = [0.2,0.1,0.05]
    x0 = 50.0

    # Simulate 100 realisations from the SSA
    P_X0 = θ -> x0 * ones(1,100)                    # initial conditions as a function of parameters
    P_T  = collect(range(0.0,10.0,length=200))      # output times
    P_X  = SimulateSSA(a,ν,P_X0,θ,P_T)              # simulate SSA

    # Simulate synthetic data (10 experiments, each with 10 observations)
    E    = 10
    X0   = θ -> x0 * ones(1,E)                      # initial conditions as a function of parameters
    T    = collect(range(1.0,10.0,step=1.0))        # observation times
    X,Y  = SimulateSSA(a,ν,X0,g_rand,θ,T,seed=10)   # simulate SSA

    # Plot 100 realisations from the SSA
    ax1  = subplot(2,4,1); ax1.set_title("Model 1")
        x1p  = ax1.plot(P_T,P_X[1,:,:],pcols[1],alpha=0.05,linewidth=2)

        # Legend and axes
        legend(x1p,["X(t)"]);
        ax1.set_yticks(50:50:250); ax1.set_xticks(0:2:10);
        ax1.set_ylim(40,260); ax1.set_xlim(-0.5,10.5); ax1.set_xticklabels([])

    # Plot synthetic data
    ax5   = subplot(2,4,5)
        ax5.plot(vcat(0.0,T),vcat(x0*ones(1,E-1),Y[1,:,2:end]),"-s",color="#333333",markersize=2,linewidth=0.5,alpha=0.3)
        y1p  = ax5.plot(vcat(0.0,T),vcat(x0,Y[1,:,1]),"-s",color="#333333",markersize=2,linewidth=0.5)

        # Legend and axes
        legend(y1p,["Y(t)"]); ax5.set_xlabel("t");
        ax5.set_yticks(50:50:250); ax5.set_xticks(0:2:10);
        ax5.set_ylim([40,260]); ax5.set_xlim(-0.5,10.5);

###############################################################################
## MODEL 2

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

    # Stoichiometries
    ν  = [-1.0  0.0 -1.0  1.0;
           0.0 -1.0  1.0 -1.0];

    # Observation process
    g_rand = (Xt,t,θ) -> Xt[1] +  θ[5] * randn()

    # "True" parameters and initial condition
    θ  = [0.1,0.2,0.2,0.5,2.0]
    x0 = [100.0,0.0]

    # Simulate 100 realisations from the SSA
    P_X0 = θ -> x0 * ones(1,100)                # initial conditions as a function of parameters
    P_T  = collect(range(0.0,10.0,length=200))  # output times
    P_X  = SimulateSSA(a,ν,P_X0,θ,P_T)          # simulate SSA

    # Simulate synthetic data (10 experiments, each with 10 observations)
    E    = 5
    X0   = θ -> x0 * ones(1,E)                      # initial conditions as a function of parameters
    T    = collect(range(1.0,10.0,step=1.0))        # observation times
    X,Y  = SimulateSSA(a,ν,X0,g_rand,θ,T,seed=20)   # simulate SSA

    # Plot 100 realisations from the SSA
    ax2   = subplot(2,4,2); ax2.set_title("Model 2")
        x1p, = ax2.plot(P_T,P_X[1,:,:],pcols[1],alpha=0.04,linewidth=2)
        x2p, = ax2.plot(P_T,P_X[2,:,:],pcols[2],alpha=0.04,linewidth=2)

        # Legend and axes
        legend([x1p,x2p],["X₁(t)","X₂(t)"]);
        ax2.set_yticks(0:20:100); ax2.set_xticks(0:2:10);
        ax2.set_ylim(-5,105); ax2.set_xlim(-0.5,10.5); ax2.set_xticklabels([])

    # Plot synthetic data
    ax6  = subplot(2,4,6)
        ax6.plot(vcat(0.0,T),vcat(x0[1]*ones(1,E-1),Y[1,:,2:end]), "-s",color="#333333",markersize=2,linewidth=0.5,alpha=0.3)
        y1p = ax6.plot(vcat(0.0,T),vcat(x0[1],Y[1,:,1]), "-s",color="#333333",markersize=2,linewidth=0.5)

        # Legend and axes
        legend(y1p,["Y(t)"]); ax6.set_xlabel("t");
        ax6.set_yticks(0:20:100); ax6.set_xticks(0:2:10);
        ax6.set_ylim(-5,105); ax6.set_xlim(-0.5,10.5);

###############################################################################
## MODEL 3

    # Epidemic model

    # X = [S,E,I]

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

    # Observation process
    g_rand = (Xt,t,θ) -> [ Xt[3] * (θ[5] + θ[6] * randn()),
                           Xt[4] * (θ[5] + θ[6] * randn()) ]

    # "True" parameters
    θ  = [0.01,0.15,0.1,10.0,0.5,0.05]

    # Simulate 100 realisations from the SSA
    P_X0 = θ -> [500 - θ[4],θ[4],10 / θ[5],10 / θ[5]] * ones(1,100) # initial conditions as a function of parameters
    P_T  = collect(range(0.0,30.0,length=200))           # output times
    P_X  = SimulateSSA(a,ν,P_X0,θ,P_T)                   # simulate SSA

    # Simulate synthetic data (1 experiment, short data out to time 10, long to time 30)
    E     = 1
    X0    = θ -> [500 - θ[4],θ[4],10 / θ[5],10 / θ[5]] * ones(1,E)   # initial conditions as a function of parameters
    Tl    = collect(range(1.0,30.0,step=1.0))            # observation times
    Xl,Yl = SimulateSSA(a,ν,X0,g_rand,θ,Tl,seed=30)       # simulate SSA
    Ts    = Tl[1:10]        # Short data is first 10 days from long data
    Ys    = Yl[:,1:10,:]    #   ""

    # Plot 100 realisations from the SSA
    ax3   = subplot(2,4,3); ax3.set_title("Model 3")
        x1p, = ax3.plot(P_T,P_X[1,:,:],pcols[1],alpha=0.02,linewidth=2)
        x2p, = ax3.plot(P_T,P_X[2,:,:],pcols[2],alpha=0.02,linewidth=2)
        x3p, = ax3.plot(P_T,P_X[3,:,:],pcols[3],alpha=0.02,linewidth=2)

        # Legend and axes
        legend([x1p,x2p,x3p],["S(t)","E(t)","I(t)"]);
        ax3.set_yticks(0:100:500); ax3.set_xticks(0:5:30);
        ax3.set_ylim(-25,525); ax3.set_xlim(-1.5,31.5); ax3.set_xticklabels([])

    # Plot synthetic data
    ax7  = subplot(2,4,7)
        y1lp, = ax7.plot(Tl,Yl[1,:,1],"-s",color=pcols[3],markersize=2,linewidth=0.5,alpha=0.3)
        y2lp, = ax7.plot(Tl,Yl[2,:,1],"-^",color="#333333",markersize=2,linewidth=0.5,alpha=0.3)
        y1sp, = ax7.plot(vcat(0.0,Ts),vcat(10.0,Ys[1,:,1]),"-s",color=pcols[3],markersize=2,linewidth=0.5)
        y2sp, = ax7.plot(vcat(0.0,Ts),vcat(10.0,Ys[2,:,1]),"-^",color="#333333",markersize=2,linewidth=0.5)

        # Legend and axes
        legend([y1sp,y2sp,y1lp,y2lp],["Y1,s(t)","Y2,s(t)","Y1,l(t)","Y2,l(t)"]); ax7.set_xlabel("t");
        ax7.set_yticks(0:50:300); ax7.set_xticks(0:5:30);
        ax7.set_ylim(-10,310); ax7.set_xlim(-1.5,31.5);

###############################################################################
## MODEL 4

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

    # Observation process
    g_rand = (Xt,t,θ) -> [Xt[1] + θ[3] * randn(),
                          Xt[3] + θ[3] * randn()]

    # "True" parameters (untransformed) and initial condition
    θ  = [3e-2,5e-4,1.0]
    x0 = [322.0,10.0,5.0]

    # Numerical parameters
    Δt = 0.2

    # Simulate 100 realisations from the SDE
    P_X0 = θ -> x0 * ones(1,100)                        # initial conditions as a function of parameters
    P_T  = collect(range(0.0,150.0,length=200))         # output times
    P_X, = SimulateSDE(α,σ,P_X0,θ,P_T,Δt)               # simulate SDE

    # Simulate synthetic data (5 experiments, each with 15 observations)
    E     = 5
    X0    = θ -> x0 * ones(1,E)                         # initial conditions as a function of parameters
    T     = collect(range(10.0,150.0,step=10.0))        # observation times
    X,Y,  = SimulateSDE(α,σ,X0,g_rand,θ,T,Δt;seed=40)   # simulate SDE

    # Plot 100 realisations from the SSA
    ax4   = subplot(2,4,4); ax4.set_title("Model 4"); ax4r = twinx()
        x1p, = ax4r.plot(P_T,P_X[1,:,:],pcols[1],alpha=0.05,linewidth=1)
        x2p, = ax4.plot(P_T,P_X[2,:,:],pcols[2],alpha=0.03,linewidth=1)
        x3p, = ax4.plot(P_T,P_X[3,:,:],pcols[3],alpha=0.03,linewidth=1)

        # Legend and axes
        legend([x1p,x2p,x3p],["β(t)","I(t)","G(t)"]);
        ax4.set_yticks(0:10:40); ax4.set_xticks(0:25:150);
        ax4r.set_yticks(300:10:340); ax4r.set_ylim(298,342)
        ax4.set_ylim(-2,42); ax4.set_xlim(-5,155); ax4.set_xticklabels([])

    #Plot synthetic data
    ax8  = subplot(2,4,8); ax8r = twinx();
        ax8r.plot(vcat(0.0,T),vcat(x0[1]*ones(1,E-1),Y[1,:,2:end]),"-^",color="#c63e30",alpha=0.2,markersize=2,linewidth=0.5)
        ax8.plot(vcat(0.0,T), vcat(x0[3]*ones(1,E-1),Y[2,:,2:end]),"k-s",alpha=0.2,markersize=2,linewidth=0.5)
        y1p, = ax8r.plot(vcat(0.0,T), vcat(x0[1],Y[1,:,1]),"-^",color="#c63e30",markersize=2,linewidth=0.5)
        y2p, = ax8.plot(vcat(0.0,T), vcat(x0[3],Y[2,:,1]),"k-s",markersize=2,linewidth=0.5)

        # Legend and axes
        legend([y1p,y2p],["Y₁(t)","Y₂(t)"]); ax8.set_xlabel("t");
        ax8.set_yticks(0:10:40); ax8.set_xticks(0:25:150);
        ax8r.set_yticks(300:10:340); ax8r.set_ylim(298,342)
        ax8.set_ylim(-2,42); ax4.set_xlim(-5,155);

## View figure 3

plt.tight_layout()
display(fig3)
