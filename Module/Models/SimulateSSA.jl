#=
#
#   SimulateSSA.jl
#
#   Simulate from the stochastic simulation algorithm (SSA) to generate synthetic
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
    function SimulateSSA(a::Function,ν::Array{Float64,2},X0::Function,g_rand::Function,θ::Array{Float64,1},T::Array{Float64,1};seed::Int=0)

Simulate observations from the stochastic simulation algorithm (SSA) for a model
specified by propensities `a` and stoichiometries `ν`. The initial condition `X0(θ)`,
is a function of the parameters, and specifies the number of experiments:

    `size(X0(θ)) = [N,E]`

where `N` is the problem dimension, and `E` is the number of experiments. Optionally
specified is an observation function `g_rand()` such that

    `Y = g_rand(X,t,θ)`

Output times must be specified in the vector `T`, which is the same length for
all experiments.

Inputs:\n
    `a`      - propensity fcn               a(X,t,θ)      : R^N × R × R^D → R^Q
    `ν`      - stoichiometric matrix        ν ∈ R^(N × Q)
    `X0_fcn` - initial condition fcn        X0(θ)         : R^d → R^(N × E)
    `g_rand` - (optional) observation fcn   g_rand(X,t,θ) : R^N × R × R^D → R^M
    `θ`      - parameter vector             θ ∈ R^E
    `T`      - output times                 T ∈ R^O
    `seed`   - (kw,optional) rng seed       seed ∈ Z

Outputs: \n
    `X`      - state variables, an N×O×E matrix, with `X[:,:,e]` is the `j`th experiment
    `Y`      - (if g_rand given) observables, an M×O×E matrix, with `X[:,:,e]` is the `j`th experiment
"""
function SimulateSSA(a::Function,ν::Array{Float64,2},X0_fcn::Function,θ::Array{Float64,1},
                        T::Array{Float64,1};seed::Int=0)

    # Obtain initial condition
    X0  = X0_fcn(θ)

    # Problem dimensions
    N,Q = size(ν)
    _,E = size(X0)
    O   = length(T)

    # Set seed for reproducability
    if seed != 0
        Random.seed!(seed)
    end

    # Initialise output
    X = zeros(N,O,E)

    # Simulate E independent trajectories
    for e = 1:E

        # Initialise
        Xc = X0[:,e]    # Initial state
        t  = 0.0        # Start time
        o  = 1          # Next observation time to save

        # Loop through time
        while t < T[end]

            # Compute rates
            R = a(Xc,t,θ)

            # Sample timestep, τ ∼ Exp(sum(R)), and update time
            τ = -log(rand()) / sum(R)
            t += τ

            # Update stored value if next event after next observation time
            while o <= length(T) && T[o] < t
                X[:,o,e] = Xc
                o += 1          # Increment to next observation
            end

            # Stop if next event is after the end time (including of next event is at infinity)
            if t > T[end] || isinf(t)
                break
            end

            # Sample event: event k happens with probability R_k / sum(R)
            I = sample(collect(1:Q),Weights(R),1)

            # Update state using stoichiometry
            Xc += ν[:,I]

        end # (while t < T[end])

    end # (for e = 1:E)

    # Restore seed
    if seed != 0
        Random.seed!()
    end

    # Return state
    return X

end

function SimulateSSA(a::Function,ν::Array{Float64,2},X0_fcn::Function,g_rand::Function,θ::Array{Float64,1},
                        T::Array{Float64,1};seed::Int=0)

    # Set seed for reproducability
    if seed != 0
        Random.seed!(seed)
    end

    # Simulate state
    X = SimulateSSA(a,ν,X0_fcn,θ,T,seed=0)

    # Problem dimensions
    _,O,E = size(X)
    M = length(g_rand(X[:,1,1],T[1],θ))

    # Obtain observables
    Y = zeros(M,O,E)
    for e = 1:E
        for o = 1:length(T)
            Y[:,o,e] .= g_rand(X[:,o,e],T[o],θ)
        end
    end

    # Restore seed
    if seed != 0
        Random.seed!()
    end

    # Return state and observables
    return X,Y

end
