#=
#
#   Euler.jl
#
#   Euler's method for use with MetropolisHastings MCMC code
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#
@doc """
    Euler!(X::Array{Float64},Δt::Float64,n::Int,t0::Float64,α::Function,θ::Array{Float64,1},N::Int,R::Int)

Applies n steps of Euler's method to simulate an ordinary differential equation
of the form

    dX/dt = α(X,t;θ)

from t0 to t0 + n × Δt where X(t0) = X.

This function modifies the input argument X.

Inputs:\n
    `X` - initial condition     1-D: R^R,   N-D: R^(N × R)
    `Δt`- time step
    `n` - number of steps
    `t0`- initial time
    `α` - RHS of ODE                α(⋅,⋅;θ) : R^N × R → R^N
    `θ` - parameter vector          θ  ∈ R^D
    `N` - number of dimensions
    `R` - number of realisations

Outputs:\n
    None - this function modifies its inputs
"""
function Euler!(X::Array{Float64},Δt::Float64,n::Int,t0::Float64,
                        α::Function,θ::Array{Float64,1},N::Int,R::Int)

    if N == 1
    # 1-D, simulate simultaneously

        ## Step through time
        @inbounds for i = 1:n

            X[:] += α(X[:],t0 + i*Δt,θ) * Δt;

        end

    else
    # N-D, simulate one-realisation at a time

        ## Loop through realisations
        @inbounds for r = 1:R

            ### Step through time
            for i = 1:n

                X[:,r] += α(X[:,r],t0 + i*Δt,θ) * Δt

            end

        end

    end

end

function Euler(X::Array{Float64},Δt::Float64,n::Int,t0::Float64,
                        α::Function,θ::Array{Float64,1},
                        N::Int,R::Int)

    X = copy(X0)

    Euler!(X,Δt,n,t0,α,θ,N,R)

    return X

end
