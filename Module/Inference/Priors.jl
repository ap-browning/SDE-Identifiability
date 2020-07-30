#=
#
#   Priors.jl
#
#   Functions for creating and using prior distributions
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       e: ap.browning@qut.edu.a
#       w: alexbrowning.me
#
=#

@doc """
    function GetLogPrior(distribution::String,params1::Array{Float64,1},params2::Array{Float64,1})

Returns log prior probability density function for the distribution specfied by
`distribution`.

    `distribution = "independent-uniform"`:

        Additional inputs:\n
            `PriorMin`  - prior lower bounds
            `PriorMax`  - prior upper bounds

"""
function GetLogPrior(distribution::String,params1::Array{Float64,1},params2::Array{Float64,1})

    # Uniform distribution
    if distribution == "independent-uniform"

        # Return log uniform prior
        LogPrior = θ -> prod((θ .> params1) .* (θ .< params2)) ? log(1/prod(params2 - params1)) : log(0.0)
        return LogPrior

    end

end

@doc """
    function MarginalPriorPDF(x::Array{Float64,2},distribution::String,params1::Array{Float64,1},params2::Array{Float64,1})

Returns prior probability density for the distribution specified by `distribution`
for parameter vector `x` where `x[i]` corresponds to the `i`th sample, and `x[j,:]` corresponds
to all samples of the `j`th variable.

    `distribution = "independent-uniform"`:

        Additional inputs:\n
            `PriorMin`  - prior lower bounds
            `PriorMax`  - prior upper bounds

"""
function MarginalPriorPDF(x::Array{Float64,1},distribution::String,params1::Array{Float64,1},params2::Array{Float64,1})

    # Uniform distribution
    if distribution == "independent-uniform"

        # Initialise output
        f = zeros(size(x))

        # Loop through variables
        for i = 1:length(params1)
             f[i] = 1 / (params2[i] - params1[i]);
        end

    end

    # Return probability density
    return f

end


@doc """
    function GetLogPrior(x::Array{Float64,2},distribution::String,params1::Array{Float64,1},params2::Array{Float64,1})

Returns `n` samples from the prior distribution specified by `distribution`

    `distribution = "independent-uniform"`:

        Additional inputs:\n
            `PriorMin`  - prior lower bounds
            `PriorMax`  - prior upper bounds

"""
function PriorSample(distribution::String,n::Int,params1::Array{Float64,1},params2::Array{Float64,1};seed::Int=0)

    # Set seed for reproducability
    if seed != 0
        Random.seed!(seed)
    end

    # Uniform distribution
    if distribution == "independent-uniform"

        D = length(params1)
        θ = (params2 - params1) .* rand(D,n) .+ params1

    end

    # Restore seed
    if seed != 0
        Random.seed!()
    end

    # Return probability density
    return θ

end
