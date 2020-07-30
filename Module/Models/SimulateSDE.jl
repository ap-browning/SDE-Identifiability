#=
#
#   SimulateSDE.jl
#
#   Simulate from a stochastic differential equation (SDE) to generate synthetic
#   data to be used for practical identifiability analysis.
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       e: ap.browning@qut.edu.a
#       w: alexbrowning.me
#
=#

@doc """
    function SimulateSDE(α::Function,σ::Function,X0_fcn::Function,g_rand::Function,θ::Array{Float64,1},T,Δt::Float64;seed::Int=0,reflect::Bool=true)

Simulates `E` synthetic experiments of a stochastic differential equation
of the form

    dX_t = `α(X_t,t;θ)`dt + `σ(X_t,t;θ)` dW_t,

and returns observations at times specified by `T` of the observed variable

    `Y_t  = g_rand(X_t)`.

The initial condition `X0(θ)`, is a function of the parameters, and specifies
the number of experiments:

    `size(X0(θ)) = [N,E]`

where `N` is the problem dimension, and `E` is the number of experiments.

Here, `{X_t,t ≥ 0}` is an N-dimensional continuous state Markov process,
`{W_t, t ≥ 0}` is a Q-dimensional Wiener process, `α(⋅,⋅;θ) : R^N × T → R^N`
is the drift function, and `σ(⋅,⋅; θ) R^N × T → R^(N×Q)` is the diffusion function.
Both `α` and `σ` are parameterised by the vector `θ ∈ R^D`.

Inputs:\n
    `α`      - drift function               α(X,t,θ)      : R^N × R × R^D → R^N
    `σ`      - diffusion function           σ(X,t,θ)      : R^N × R × R^D → R^(N × Q)
    `X0`     - initial condition fcn        X0(θ)         : R^d → R^(N × E)
    `g_rand` - (optional) observation fcn   g_rand(X,t,θ) : R^N × R × R^D → R^M
    `θ`      - parameter vector             θ  ∈ R^D
    `T`      - output times                 T  ∈ R^O or Ti ∈ Z^O where T = Δt × Ti
    `Δt`     - timestep                     Δt ∈ R
    `seed`   - (kw,optional) rng seed       seed ∈ Z
    `reflect`- (kw,optional) reflect at 0   true/false (default: true)

Outputs: \n
    `X`      - state variables, an N×O×E matrix, with `X[:,:,e]` is the `j`th experiment
    `Y`      - (if g_rand given) observables, an M×O×E matrix, with `X[:,:,e]` is the `j`th experiment
    `Ti`     - (if T given as Float64) observation times in units of Δt
"""
function SimulateSDE(α::Function,σ::Function,X0_fcn::Function,θ::Array{Float64,1},
                        Ti::Array{Int,1},Δt::Float64;seed::Int=0,reflect::Bool=true)

    # Obtain initial condition
    X0  = X0_fcn(θ)

    # Problem dimensions
    N,Q = size(σ(X0[:,1],0.0,θ))
    _,E = size(X0)
    O   = length(Ti)

    # Set seed for reproducability
    if seed != 0
        Random.seed!(seed)
    end

    # Initialise output
    X = zeros(N,O,E)

    # Simulate E independent trajectories
    for e = 1:E

        # Initialise
        Xprev = X0[:,e]     # Initial state
        Tis = 0             # Initialise time (units of Δt)

        # Loop through observation points
        for o = 1:O

            # Move realisation forward from Xprev at time Tis * Δt using Euler Maruyama algorithm
            X[:,o,e] = EulerMaruyama(Xprev,Δt,Ti[o]-Tis,Tis*Δt,α,σ,θ,N,Q,1,reflect=reflect)

            # Update time and state
            Tis = Ti[o]
            Xprev = X[:,o,e]

        end # (for o = 1:O)

    end # (for e = 1:E)

    # Restore seed
    if seed != 0
        Random.seed!()
    end

    # Return state
    return X

end

## Not given T as set of indices
function SimulateSDE(α::Function,σ::Function,X0_fcn::Function,θ::Array{Float64,1},
        T,Δt::Float64;seed::Int=0,reflect::Bool=true)

    # Time in units of Δt
    Ti = Int.(floor.(T / Δt))

    # Simulate state
    X = SimulateSDE(α,σ,X0_fcn,θ,Ti,Δt,seed=seed,reflect=reflect)

    # Return state and time in units of Δt
    return X,Ti

end

## Given observation function
function SimulateSDE(α::Function,σ::Function,X0_fcn::Function,g_rand::Function,θ::Array{Float64,1},
                Ti::Array{Int,1},Δt::Float64;seed::Int=0,reflect::Bool=true)

    # Set seed for reproducability
    if seed != 0
        Random.seed!(seed)
    end

    # Simulate state
    X = SimulateSDE(α,σ,X0_fcn,θ,Ti,Δt,seed=0,reflect=reflect)

    # Problem dimensions
    _,O,E = size(X)
    M = length(g_rand(X[:,1,1],Δt * Ti[1],θ))

    # Obtain observables
    Y = zeros(M,O,E)
    for e = 1:E
        for o = 1:length(Ti)
            Y[:,o,e] .= g_rand(X[:,o,e],Δt*Ti[o],θ)
        end
    end

    # Restore seed
    if seed != 0
        Random.seed!()
    end

    # Return state and observables
    return X,Y

end

## Given observation function, but not T as set of indices
function SimulateSDE(α::Function,σ::Function,X0_fcn::Function,g_rand::Function,θ::Array{Float64,1},
            T,Δt::Float64;seed::Int=0,reflect::Bool=true)

    # Time in units of Δt
    Ti = Int.(floor.(T / Δt))

    # Simulate state and observations
    X,Y = SimulateSDE(α,σ,X0_fcn,g_rand,θ,Ti,Δt,seed=seed,reflect=reflect)

    # Return state, observables, and time in units of Δt
    return X,Y,Ti

end
