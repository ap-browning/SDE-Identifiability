#=

    SEIR_Compare.jl

    Compare each closure for the SEIR model. Results correspond to Figure S1.

=#

using DifferentialEquations
using PyPlot
include("SEIR_MomentEqs.jl")
include("LabelMaker.jl")
include("../Results/IdentifiabilityProjectOptions.jl")

# Parameters
θ  = [0.01,0.15,0.1]
μ  = 0.5
e₀ = 10.0

# Initial condition (O(1))
X₀ = [500.0 - e₀,e₀,10/μ,10/μ]

# Initial condition (O(2))
m₀  = i -> prod(X₀.^i)
I   = [[1,0,0,0],
       [0,1,0,0],
       [0,0,1,0],
       [0,0,0,1],
       [2,0,0,0],
       [0,2,0,0],
       [0,0,2,0],
       [0,0,0,2],
       [1,1,0,0],
       [1,0,1,0],
       [1,0,0,1],
       [0,1,1,0],
       [0,1,0,1],
       [0,0,1,1]]
M₀  = m₀.(I)

# Range
tspan = (0.0,30.0)

# Time-step for Tanaka method
Δt  = 0.001

# Plotting times
T   = range(tspan...,step=Δt)

# Solve and store a solution using each closure
RHS  = [SEIR_ODE, SEIR_MeanField, SEIR_Pairwise, SEIR_Gaussian]
solT = Array{Any,1}(undef,4)
for c = 1:4

    ic   = c == 1 ? X₀ : M₀

    # Solve using Tsit5
    if c != 2

        sol     = solve(ODEProblem(RHS[c],ic,tspan,θ),Tsit5())
        solT[c] = hcat(sol.(T)...)

    # Solve using Patankar algorithm
    else

        solT[c] = zeros(length(ic),length(T))
        solT[c][:,1] = ic

        for i = 2:length(T)

            mt,t  = solT[c][:,i-1],T[i-1]

            P,D = SEIR_MeanField(mt,θ,t)
            solT[c][:,i] = @. (mt + Δt * P) / (1 + Δt * D / mt)

        end

    end

end

# Plot

# Styles for each closure
styles = ["b","r:","g--","y-."]
labels = ["ODE","Mean-Field","Pairwise","Gaussian"]

# Plot O(1)
fig,axs = subplots(4,4,figsize=(7.2,6.1)) # First order moments

for c = 1:4
    for i = 1:(c == 1 ? 4 : 14)

        # First order
        if any(i .== 1:4)
            row,col = 1,i
        elseif any(i .== 5:8)
            row,col = 2,i-4
        elseif any(i .== 9:11)
            row,col = 3,i-8
        else
            row,col = 4,i-11
        end

        axs[row,col].plot(T[1:10:end],solT[c][i,1:10:end],styles[c])
        axs[row,col].set_title(LabelMaker(I[i]))
        axs[row,col].set_xticks(0.0:10.0:30.0)

        if row == 1
            axs[row,col].set_ylim((-20.0,620.0))
            axs[row,col].set_yticks(0.0:200.0:600.0)
        end

    end
end

axs[1,1].legend(labels)
plt.tight_layout()
display(fig)
