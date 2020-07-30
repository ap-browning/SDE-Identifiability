#=
#
#   PosteriorPredictive.jl
#
#   Produce posterior predictive quantiles based on MCMC samples and model
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#

@doc """
    PosteriorPredictiveQuantiles(Θ,q,a,ν,X0_fcn,g_rand,T,Δt,reps;...)

Produce posterior predictive quantiles using an SDE CLE described by
propensities `a` and stoichiometries `ν`. MCMC results are given as `Θ`, which
are resampled (with replacement) to produce `reps` SDE trajectories.

Inputs:\n
    `Θ`      - MCMC output
    `q`      - vector of quantiles
    `a`      - propensity fcn               a(X,t,θ)      : R^N × R × R^D → R^Q
    `ν`      - stoichiometric matrix        ν ∈ R^(N × Q)
    `X0_fcn` - initial condition fcn        X0(θ)         : R^d → R^(N × E)
    `g_rand` - (optional) observation fcn   g_rand(X,t,θ) : R^N × R × R^D → R^M
    `T`      - output times                 T  ∈ R^O
    `Δt`     - time step                    Δt ∈ R
    `reps`   - number of realisations       reps ∈ Z
    `burnin` - (optional, default: 3000)    discard samples as burn-in
    `skip`   - (optional, default: 1)       thin chain
    `reflect` - whether to keep positive state values by reflection

Outputs: \n
    `Q`      - an M × O × length(q) matrix of quantiles
"""
function PosteriorPredictiveQuantiles(Θ::Array{Float64,3},q::Array{Float64,1},
                a::Function,ν::Array{Float64,2},X0_fcn::Function,g_rand::Function,
                T::Array{Float64,1},Δt::Float64,reps::Int64;
                burnin::Int64=3000,skip::Int64=1,reflect::Bool=true)

    # Problem size
    C,D,S = size(Θ)
    N,= size(ν)
    M = length(g_rand(X0_fcn(Θ[1,:,1])[:,1],0.0,Θ[1,:,1]))

    # Drift and diffusion function
    α = (X,t,θ) -> ν*a(X,t,θ)                   # Drift function
    σ = (X,t,θ) -> ν*diagm(a(X,t,θ).^(1/2))     # Diffusion function

    # Parameters to resample
    indices = burnin+2:skip:S
    ns_each = length(indices)
    samples = zeros(D,C*ns_each)
    for c = 1:C
        samples[:,ns_each*(c-1)+1:ns_each*c] = Θ[c,:,indices]
    end
    _,NS = size(samples)

    # Initialise outputs
    Y = zeros(M,length(T),reps)                 # Store all trajectories
    Q = zeros(M,length(T),length(q))            # Store quantiles

    # Loop through reps
    for r = 1:reps

        ## Resample parameters
        i = rand(1:NS)
        θ = samples[:,i]

        ## Simulate SDE
        Xt,Yt,Ti = SimulateSDE(α,σ,X0_fcn,g_rand,θ,T,Δt,reflect=reflect)

        ## Store SDE trajectories
        Y[:,:,r] = Yt

    end

    # Compute quantiles
    for k = 1:M

        ## Loop through time
        for t = 1:length(T)

            ### Compite `q` quantiles of Y[k,t,:]
            Q[k,t,:] = quantile(Y[k,t,:],q)

        end

    end

    # Return quantiles at each time-point
    return Q

end
