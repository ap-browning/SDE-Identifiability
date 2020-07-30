#=
#
#   EulerMaruyama.jl
#
#   Euler Maruyama algorithm for use with Pseudo Marginal MCMC code
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#
@doc """
    EulerMaruyama!(X::Array{Float64},Δt::Float64,n::Int,t0::Float64,α::Function,σ::Function,θ::Array{Float64,1},N::Int,Q::Int,R::Int;reflect::Bool=true)

Applies n steps of the Euler Maruyama algorithm to simulate a stochastic
differential equation of the form

    dX_t = α(X_t,t;θ)dt + σ(X_t,t;θ)dW_t

from t0 to t0 + n × Δt where X(t0) = X.

This function modifies the input argument X.

Inputs:\n
    `X` - initial condition     1-D: R^R,   N-D: R^(N × R)
    `Δt`- time step
    `n` - number of steps
    `t0`- initial time
    `α` - drift function                α(⋅,⋅;θ) : R^N × R → R^N
    `σ` - diffusion function            σ(⋅,⋅;θ) : R^N × R → R^(N × Q)
    `θ` - parameter vector              θ  ∈ R^D
    `N` - number of dimensions
    `Q` - number of Wiener processes
    `R` - number of realisations
    `reflect` - whether to keep positive state values by reflection

Outputs:\n
    None - this function modifies its inputs
"""
function EulerMaruyama!(X::Array{Float64},Δt::Float64,n::Int,t0::Float64,
                        α::Function,σ::Function,θ::Array{Float64,1},
                        N::Int,Q::Int,R::Int;reflect::Bool=true)

    if N == 1
    # 1-D, simulate simultaneously

        ### Precompute Wiener-increments
        ΔW = sqrt(Δt) * randn(n,R,Q)

        ### Step through time
        for i = 1:n

            X[:] += α(X[:],t0 + (i-1)*Δt,θ) * Δt .+ sum(σ(X[:],t0 + (i-1)*Δt,θ) .* ΔW[i,:,:],dims=2)

            #### Reflect if necessary
            if reflect
                X[:] = abs.(X[:])
            end

        end

    else
    # N-D, similate one realisation at a time

        ## Loop through realisations
        for r = 1:R

            ### Precompute Wiener-increments
            ΔW = sqrt(Δt) * randn(Q,n)

            ### Step through time
            for i = 1:n
                X[:,r] += α(X[:,r],t0 + (i-1)*Δt,θ) * Δt + σ(X[:,r],t0 + (i-1)*Δt,θ) * ΔW[:,i]

                #### Reflect if necessary
                if reflect
                    X[:,r] = abs.(X[:,r])
                end
            end

        end

    end

end

function EulerMaruyama(X0::Array{Float64},Δt::Float64,n::Int,t0::Float64,
                        α::Function,σ::Function,θ::Array{Float64,1},
                        N::Int,Q::Int,R::Int;reflect::Bool=true)

    X = copy(X0)

    EulerMaruyama!(X,Δt,n,t0,α,σ,θ,N,Q,R,reflect=reflect)

    return X

end
