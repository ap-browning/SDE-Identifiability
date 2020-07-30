#=
#
#   MetropolisHastings.jl
#
#   Perform Metropolis-Hastings MCMC using Pseudo Marginal MCMC (or calculating
#   the likelihood directly for a deterministic model)
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#
@doc """
    PseudoMarginalMH(α::Function,σ::Function,g::Function,
                              X::Array{Float64,3},Ti::Array{Int64,1},x0::Array{Float64},
                              Δt::Float64,R::Int,θ0::Array{Float64,1},Σ::Array{Float64,2},
                              p0::Function,M::Int;dep::Bool=true,reflect::Bool=true)

Use Pseudo Marginal methods (Warne 2020) to approximate the likelihood function
and perform Metropolis Hastings MCMC for data, X, from the SDE of the form

    dX_t = α(X_t,t;θ)dt + σ(X_t,t;θ)dW_t

subject to observations Y = g(X,t).

For exact observations (only N = 1 currently supported), g may be omitted.

Inputs:\n
    `α`        - drift function              α(⋅,⋅;θ) : R^N × R → R^N
    `σ`        - diffusion function          σ(⋅,⋅;θ) : R^N × R → R^(N × M)
    `g`        - observation process pdf     g(⋅,⋅)   : R^N × R → R^+
    `Y`        - observations                R^(N × O × E)
    `Ti`       - time points (units of Δt)   Z^O
    `X0`       - initial condition           R^(N × E)
    `Δt`       - Euler Maruyama timestep
    `R`        - PseudoMarginal realisations
    `θ0`       - chain initial condition
    `Σ`        - proposal covariance matrix (MvNormal)
    `log_π`    - prior                       p(⋅)    : R^d → p ∈ [0,1]
    `S`        - chain length
    `C`        - number of chains
    `reflect`  - (optional, default: true)  ensure positivity in SDE solutions using reflecting BCs
    `seed`     - (optional)                 input RNG seed

Outputs:\n
    `Θ`        - MCMC sequence         R^(length(θ) × M)

"""
# Noisy observations, α and σ given
function PseudoMarginalMH(α::Function,σ::Function,X0::Function,g_lpdf::Function,
                          Y::Array{Float64,3},Ti::Array{Int64,1},
                          Δt::Float64,R::Int,θ0::Array{Float64,2},Σ::Array{Float64,2},
                          log_π::Function,S::Int,C::Int;reflect::Bool=true,seed::Int=0)

    # Problem dimensions
    _,O,E = size(Y)
    N,Q   = size(σ(X0(θ0[:,1])[:,1],0.0,θ0[:,1]))

    # Euler Maruyama wrapper (move Xs n steps forward from time t)
    function EM_Sim!(θ::Array{Float64,1},t::Float64,Xs::Array{Float64},n::Int)
        EulerMaruyama!(Xs,Δt,n,t,α,σ,θ,N,Q,R,reflect=reflect)
    end

    # Approximate log-likelihood with particle filter
    LogLike = θ -> PMLogLikeParticleFilter(θ[:],X0(θ),EM_Sim!,g_lpdf,Y,Ti,Δt,R)

    # Perform MCMC and return
    return MetropolisHastings(LogLike,θ0,Σ,log_π,S,C,seed=seed)

end

# Noisy observations, a and ν given (CLE version)
function PseudoMarginalMH(a::Function,ν::Array{Float64,2},X0::Function,g_lpdf::Function,
                          Y::Array{Float64,3},Ti::Array{Int64,1},
                          Δt::Float64,R::Int,θ0::Array{Float64,2},Σ::Array{Float64,2},
                          log_π::Function,S::Int,C::Int;reflect::Bool=true,seed::Int=0)

    # Get problem dimensions
    _,O,E = size(Y)
    N,Q = size(ν)

    # Euler Maruyama wrapper (move Xs n steps forward from time t)
    function EM_Sim!(θ::Array{Float64,1},t::Float64,Xs::Array{Float64},n::Int)
        α = (X,t,θ) -> ν*a(X,t,θ)                   # Drift function
        σ = (X,t,θ) -> ν*diagm(a(X,t,θ).^(1/2))     # Diffusion function
        EulerMaruyama!(Xs,Δt,n,t,α,σ,θ,N,Q,R,reflect=reflect)
    end

    # Approximate log-likelihood with particle filter
    LogLike = θ -> PMLogLikeParticleFilter(θ[:],X0(θ),EM_Sim!,g_lpdf,Y,Ti,Δt,R)

    # Perform MCMC and return
    return MetropolisHastings(LogLike,θ0,Σ,log_π,S,C,seed=seed)

end


# ODE inference
function DeterministicMH(α::Function,X0_fcn::Function,g_lpdf::Function,
                          Y::Array{Float64,3},Ti::Array{Int64,1},
                          Δt::Float64,θ0::Array{Float64,2},Σ::Array{Float64,2},
                          log_π::Function,S::Int,C::Int;seed::Int=0)

    # Get problem dimensions
    _,O,E = size(Y)
    N,_ = size(X0_fcn(θ0))

    # Log Likelihood function
    function LogLike(θ)

        ## Initial condition
        X0 = X0_fcn(θ)

        ## Initialise
        LL = 0.0

        ## Loop through experiments
        for e = 1:E

            ## Initialise
            Xs = X0[:,e]
            Tis = 0

            ## Loop through observations (assume independent for ODE)
            for o = 1:O

                ### Simulate forward using Forward-Euler
                Euler!(Xs,Δt,Ti[o]-Tis,Tis*Δt,α,θ,N,1)

                ### Move to next observation
                Tis = Ti[o]

                ### Update likelihood based on g
                LL += g_lpdf(Y[:,o,e],Xs,Tis*Δt,θ)

            end # (for i = 1:O)

        end # (for e = 1:E)

        # Return log likelihood estimate
        return LL

    end

    # Perform MCMC and return
    return MetropolisHastings(LogLike,θ0,Σ,log_π,S,C,seed=seed)

end

# ODE inference (CLE form)
function DeterministicMH(a::Function,ν::Array{Float64,2},X0_fcn::Function,g_lpdf::Function,
                          Y::Array{Float64,3},Ti::Array{Int64,1},
                          Δt::Float64,θ0::Array{Float64,2},Σ::Array{Float64,2},
                          log_π::Function,S::Int,C::Int;seed::Int=0)

    # Construct ODE
    α = (X,t,θ) -> ν*a(X,t,θ)

    # Perform MCMC and return
    return DeterministicMH(α,X0_fcn,g_lpdf,Y,Ti,Δt,θ0,Σ,log_π,S,C,seed=seed)

end

@doc """
    MetropolisHastings(LogLike::Function,θ0::Array{Float64,2},Σ::Array{Float64,2},log_π::Function,S::Int,C::Int)

Perform `C` MCMC chains of `S` iterations each using the Metropolis Hastings algorithm
with a multivariate normal prior, with covariance matrix `Σ`, initated at `θ0`


Inputs:\n
    `LogLike`   - log likelihood function       LogLike(⋅) : R^D → R
    `θ0`        - initial chain state           R^(D × C)
    `Σ`         - proposal covariance matrix    R^(D × D)
    `log_π`     - prior distribution            log_π(⋅)   : R^D → R
    `S`         - chain length
    `C`         - number of chains
    `seed`      - (optional) RNG seed

Outputs:\n
    `Θ`         - MCMC sequence
    `L`         - LogLikelihood estimate

"""
function MetropolisHastings(LogLike::Function,θ0::Array{Float64,2},Σ::Array{Float64,2},
                            log_π::Function,S::Int,C::Int;seed::Int=0)


    # Set seed for reproducability
    if seed != 0
        Random.seed!(seed)
    end

    # Decompose covariance matrix to sample
    G = cholesky(Σ).L

    # Problem dimension
    D = length(θ0[:,1])

    # Initialise
    Θ = zeros(C,D,S+1)
    L = zeros(C,S+1)

    # Loop through chains
    @threads for c = 1:C

        ## Initialise chain
        θm          = θ0[:,c]       # Initial chain starting point
        Θ[c,:,1]    = θm
        LLm         = LogLike(θm)   # Initial log likelihood
        LPm         = log_π(θm)     # Initial prior
        L[c,1]      = LLm

        ## Iterate
        for i = 2:S+1

            ### Propose, θs ∼ MvNormal(θm,Σ)
            θs = θm + G * randn(D,1)

            ### Prior density at proposal
            LPs = log_π(θs[:])

            ### Reject, unless accept
            accept = false

            ### Proceed if prior density non zero
            if !isinf(LPs)

                #### Likelihood at proposal
                LLs = LogLike(θs[:])

                #### Metropolis-Hastings acceptance probability (symmetric proposal)
                α = min(0.0,LLs + LPs - LLm - LPm)
                if log(rand()) < α
                    accept = true
                end

            end

            ### Accept, θm ← θs, else do nothing
            if accept
                θm = θs
                LLm = LLs
                LPm = LPs
            end

            Θ[c,:,i] = θm
            L[c,i] = LLm

        end # (for i = 2:S+1)

    end # (for c = 1:C)

    # Restore seed
    if seed != 0
        Random.seed!()
    end

    # Return MCMC chains and likelihood function
    return Θ,L

end




# function PseudoMarginalLogLike(a::Function,ν::Array{Float64,2},g::Function,
#                           Y::Array{Float64,3},Ti::Array{Int64,1},X0::Function,
#                           Δt::Float64,R::Int,θ::Array{Float64,1};dep::Bool=true,reflect::Bool=true)
#
#     N,Q = size(ν)
#     _,O,E = size(Y)
#
#     # Euler Maruyama wrapper (move Xs n steps forward from time t)
#     function EM_Sim!(θ,t,Xs,n)
#         α = (X,t,θ) -> ν*a(X,t,θ)
#         σ = (X,t,θ) -> ν*diagm(a(X,t,θ).^(1/2))
#         EulerMaruyama!(Xs,Δt,n,t,α,σ,θ,N,Q,R,reflect=reflect)
#     end
#
#     return PMLogLikeParticleFilter(θ[:],EM_Sim!,g,Y,Ti,X0(θ),Δt,R,N,Q,O,E,reflect=reflect)
#
# end


# Exact observations version (only 1D currently)
# function PseudoMarginalMH(α::Function,σ::Function,
#                           X::Array{Float64,3},Ti::Array{Int64,1},X0::Array{Float64},
#                           Δt::Float64,R::Int,θ0::Array{Float64,2},Σ::Array{Float64,2},
#                           LogPrior::Function,M::Int,C::Int;dep::Bool=true,reflect::Bool=true)
#
#     N,O,E = size(X)
#
#     if N != 1
#         error("Pseudo Marginal MH with exact observations currently only supports single state problems")
#     end
#
#     LogLike = θ -> PMLogLikeKDE(θ[:],α,σ,X,Ti,X0,Δt,R,N,O,E,dep=dep,reflect=reflect)
#
#     return MetropolisHastings(LogLike,θ0,Σ,LogPrior,M,C)
#
# end
