#=
#
#   Diagnostics.jl
#
#   Compute R̂ and effective sample size of Gelman et al. (2014)
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#

@doc """
    Diagnostics(Θ::Array{Float64,3};burnin::Int=0)

Compute the R̂ and S_eff diagnostics of the MCMC results `Θ`

Inputs: \n
    `Θ`        - MCMC output  R^(C × N × M)
    `burnin`   - (optional, default = 0) number of samples to discard

Outputs: \n
    `R̂`        - Estimate of the ratio of between-chain and within-chain variance
    `Seff`     - Effective sample size

"""
function Diagnostics(Θfull::Array{Float64,3};burnin::Int=0)

    Θ = Θfull[:,:,burnin+2:end]

    C,D,S = size(Θ)

    # Outputs and intermedates
    R̂       = zeros(D)
    Seff    = zeros(D)

    # Calculate diagnostic for each dimension
    for k = 1:D

        Ψ = Θ[:,k,:]
        Ψmeans = mean(Ψ,dims=2) # Chain means
        Ψvars  = var(Ψ,dims=2)  # Chain variances

        # Between-chain variance
        B      = S / (C - 1) * sum((Ψmeans .- mean(Ψ)).^2)

        # Within-chain variance
        W      = 1 / C * sum(Ψvars)
        varΨ   = (S - 1)/S * W + 1 / S * B
        R̂[k]   = sqrt(varΨ / W)

        # Computer variogram
        V      = zeros(S)
        for t = 1:S
            V[t] = 1 / (C*(S-t)) * sum((Ψ[:,t+1:end] - Ψ[:,1:S-t]).^2)
        end

        # Approximate correlation
        ρ̂ = 1 .- V ./ (2*varΨ)

        # Find minimum T such that ρ̂[T+1] + ρ̂[T+2] < 0
        T = 1
        for t = 1:2:S-2
            if ρ̂[t+1] + ρ̂[t+2] < 0
                T = t
                break;
            end
        end

        # Compute effective sample size
        Seff[k] = S * C / (1 + 2 * sum(ρ̂[1:T]))

    end

    return R̂,Seff

end
