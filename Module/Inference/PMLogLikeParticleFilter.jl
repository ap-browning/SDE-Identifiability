#=
#
#   PMLogLikeParticleFilter.jl
#
#   Pseudo Marginal approximation to the likelihood function using a bootstrap
#   particle filter.
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#

@doc """
    function PMLogLikeParticleFilter(θ::Array{Float64,1},X0::Array{Float64,2},EM_Sim!::Function,g_lpdf::Function,Y::Array{Float64,3},Ti::Array{Int64,1},Δt::Float64,R::Int)

Compute pseudo-marginal estimate of log-likelihood function at `θ` using a bootstrap
particle filter with `R` particles. SDE function must be given as

    `EM_Sim!(θ,t,Xs,n)`

where `θ` is the parameter vector, `t` is the current time, `Xs` is the current
state and `n` is the number of steps to move forward.

Observation probability density function, `g_lpdf(Yt,Xt,t,θ)`, must also be provided
where `g_lpdf(Yt,Xt,t,θ)` = g(Y|X,t,θ).

Inputs: \n
    `θ`        - parameter vector                       R^D
    `X0`       - initial conditions                     R^(N × E)
    `EM_Sim!`  - Euler-Maruyama fuction wrapper (see description)
    `g_lpdf`    - pdf of observation function            g(Y,X,θ,t)    : R^N × R × R^D → R^+
    `Y`        - data                                   R^(N × O × E)
    `Ti`       - time points (units of Δt)              Z^O
    `Δt`       - Euler-Maruyama timestep
    `R`        - number of particles in particle filter

Outputs: \n
    `LLhat`    - Pseudo Marginal estimate of log likelihood

"""
function PMLogLikeParticleFilter(θ::Array{Float64,1},X0::Array{Float64,2},
                        EM_Sim!::Function,g_lpdf::Function,
                        Y::Array{Float64,3},Ti::Array{Int64,1},
                        Δt::Float64,R::Int)

    # Problem dimensions
    O   = length(Ti)
    N,E = size(X0)

    # Initialise
    LLhat = 0.0

    # Loop through experiments
    for e = 1:E

        ## Initalise
        Xs  = X0[:,e] * ones(1,R)   # Initialise R particles with identical initial condition
        Tis = 0                     # Initial time in units of Δt

        ## Loop through observations
        for o = 1:O

            ### Simulate R realisations from t = Tis * Δt forward by Ti[o] - Tis units of Δt
            Xprev = copy(Xs)
            EM_Sim!(θ,Tis*Δt,Xs,Ti[o]-Tis)

            ### Update time (units of Δt)
            Tis = Ti[o]

            ### Calculate particle weights
            LW = zeros(R)
            for r = 1:R
                LW[r] = g_lpdf(Y[:,o,e],Xs[:,r],Tis*Δt,θ)  # W_r = g(Y_obs|X_r)
            end

            ### Normalise by maximum (maximum has a weight of unity)
            MaxLW = maximum(LW)         # Maximum log weight

            ### Error if NaN encountered
            if isnan(MaxLW)
                error("NaN encountered in likelihood function.")
            end

            ### Break if likelihood vanishes (largest weight is zero)
            if isinf(MaxLW)
                return -Inf
            end

            Wt    = exp.(LW .- MaxLW)   # Normalised weights by maximum

            ### Loglikelihood contribution for observation `o` of experiment `e`
            LL_oe = log(sum(Wt)) + MaxLW - log(R)

            ### Add log(sum(W) / R) to likelihood function
            LLhat += LL_oe

            ### Break if likelihood vanishes
            if isinf(LLhat)
                return -Inf
            end

            ### Resample particles for next iteration
            I = sample(collect(1:R),Weights(Wt),R,replace=true)   # Sample particle indices (with replacement)

            Xs  = Xs[:,I]                                         # Update particles based on samples


        end # (for o = 1:O)

    end # (for e = 1:E)

    # Return log-likelihood estimate
    return LLhat

end
