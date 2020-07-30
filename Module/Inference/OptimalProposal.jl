#=
#
#   OptimalProposal.jl
#
#   Compute optimal proposal using the scaling law of
#   Roberts and Rosenthal (2001)
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       e: ap.browning@qut.edu.a
#       w: alexbrowning.me
#
=#

@doc """
    OptimalProposal(Θ::Array{Float64,3};burnin::Int=3000,skip::Int=1)

Compute optimal proposal using the scaling law of Roberts and Rosenthal (2001)

Inputs: \n
    `Θ`        - MCMC output  R^(C × N × M)
    `burnin`   - (optional, default = 3000) number of samples to discard
    `skip`     - (optional, default = 1) number of samples to skip when thinning chani

Outputs: \n
    `Σ̂`        - Tuned proposal kernel covariance of

"""
function OptimalProposal(Θ::Array{Float64,3};burnin::Int=3000,skip::Int=1)

    # Get problem dimensions
    C,D,Sp  = size(Θ)
    S       = Sp - 1

    # Indices to include (discard based on burnin and skip)
    indices = burnin+1:skip:Sp

    # Number from each chain
    neach   = length(indices)

    # Pooled samples
    samples = zeros(D,C*neach)

    # Loop through chains and pool samples
    for c = 1:C
        samples[:,neach*(c-1)+1:neach*c] = Θ[c,:,indices]
    end

    # Return optimally scaled proposal
    return 2.38^2 / D * cov(samples')

end
