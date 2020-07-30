#=
#
#   CredibleIntervals.jl
#
#   Approximate posterior credible intervals
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#
@doc """
    CredibleIntervals(Θ::Array{Float64,3}; method::String="quantiles",
            q::Array{Float64,1}=[0.025,0.975],burnin::Int=2000,skip::Int=20)

Approximate 95% (default) posterior credible intervals using MCMC samples Θ

Inputs: \n
    `Θ`        - MCMC output  R^(C × N × M)
    `method`   - (optional, default = "quantiles")
    `q`        - (optional, default = [0.025,0.975]) [for "quantiles" method only]
    `burnin`   - (optional, default = 3000)
    `skip`     - (optional, default = 20)

Outputs: \n
    `Q`        - Credible interval estimate

"""
function CredibleIntervals(Θfull::Array{Float64,3};method::String="quantiles",
                q::Array{Float64,1}=[0.025,0.975],burnin::Int=0,skip::Int=1)

    # Discard burn-in from chain
    Θ = Θfull[:,:,burnin+2:end]

    # Problem dimensions
    C,D,M = size(Θ)

    # Get samples after burn-in and thinning
    indices = burnin+1:skip:M
    ns_each = length(indices)
    samples = zeros(D,C*ns_each)
    for c = 1:C
        samples[:,ns_each*(c-1)+1:ns_each*c] = Θ[c,:,indices]
    end

    # Calculate credible interval
    Q = zeros(D,2)
    for k = 1:D

        if method == "quantiles"
            Q[k,:] = quantile(samples[k,:],q)
        else
            error("Unsupported method.")
        end

    end

    # Return credible intervals
    return Q

end
