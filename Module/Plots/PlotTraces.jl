#=
#
#   PlotTraces.jl
#
#   Produce MCMC trace plots
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#
@doc """
    PlotTraces(Θ::Array{Float64,3},L::Array{Float64,2};...)

Produce MCMC trace plots

Inputs:\n
    `Θ`             - MCMC results
    `fig`           - (optional)                    handle to figure to plot
    `PMin`          - (optional)                    minimum axis limits for each variable (prior minimum)
    `PMax`          - (optional)                    maximum axis limits for each variable (prior maximum)
    `LLElim`        - (optional)                    limits for log-likelihood plot
    `LLEticks`      - (optional)                    axis ticks for log-likelihood plot
    `truevals`      - (optional)                    true values to plot
    `linewidth`     - (optional, default: 1)        linewidth for traces
    `showfig`       - (optional, default: true)     show figure once produced
    `σlast`         - (optional, default: true)     σ_err variable last
    `labels`        - (optional)                    variable names for labels
    `singlecol`     - (optional, default: false)    plot traces in a single column
    `c_alpha`       - (optional, default: 0.3)      transparency for all chains but the first

Outputs:\n
    `fig`      - handle to figure
"""
function PlotTraces(Θ::Array{Float64,3},L::Array{Float64,2};
            fig=0,PMin=0,PMax=0,LLElim=0,LLEticks=0,truevals=0,linewidth=1,
            showfig=true,σlast=true,labels=0,singlecol=false,c_alpha=0.3)

    ## Create new figure?
    if fig == 0
        fig = figure()
    else
        fig = figure(fig.number)
        clf()
    end

    ## Problem size
    C,D,M = size(Θ)
    M -= 1

    ## Iteration to plot (x-axis)
    iter = (0:M) / 1e3

    ## Where to plot each variable
    if singlecol

        rows,cols = D + 1,1
        PlotIndex = k -> k + 1

    else

        cols = 2
        if iseven(D)
        # If double column, and D even, plot as follows:
            # L  .
            # v1 v2
            # ⋮

            PlotIndex = k -> 2 + k
            rows = Int(1 + D / 2)
        else
        # If double column, and D odd, plot as follows with σ up the top
            # L  σerr
            # v1  v2
            # ⋮

            if σlast
                PlotIndex = k -> (k == D) ? 2 : 2 + k
            else
                PlotIndex = k -> 1 + k
            end
            rows = Int(ceil(D / 2))

        end

    end

    # Axis for each variable and log-likelihood
    axs = Array{Any,1}(undef,D+1)

    # Loop through chains
    for c = 1:C

        ## Full transparency for first chain
        if c == 1
            alpha = 1.0
        else
            alpha = c_alpha
        end

        ## Plot Log Likelihood estimate
        if c == 1
            axs[1] = subplot(rows,cols,1)
        end
        axs[1].plot(iter,L[c,:],linewidth=linewidth,alpha=alpha)

            ### Hide xtick labels
            axs[1].set_xticklabels([])

            ### Set y label
            axs[1].set_ylabel("LLE")

            ### Axis limits and ticks
            if LLElim != 0
                axs[1].set_ylim(LLElim)
            end
            if LLEticks != 0
                axs[1].set_yticks(LLEticks)
            end

        ## Plot variables
        for k = 1:D

            ### Get plot index and plot trace
            PI      = PlotIndex(k)
            if c == 1
                axs[1+k] = subplot(rows,cols,PI)
            end
            axs[1+k].plot(iter,Θ[c,k,:],linewidth=linewidth,alpha=alpha)

            ### Nice limits and ticks
            if PMin != 0
                if PMax != 0
                    axs[1+k].set_ylim([PMin[k],PMax[k]])
                    axs[1+k].set_yticks([PMin[k],0.5*(PMin[k] + PMax[k]),PMax[k]])
                end
            end

            ### If double column, plot y axis ticks on the right
            if iseven(PI) && !singlecol
                axs[1+k].yaxis.tick_right()
                axs[1+k].yaxis.set_label_position("right")
            end

            ### Xticklabels if on the bottom of the figure
            if PI <= rows*cols - cols || (singlecol && PI != rows)
                axs[1+k].set_xticklabels([])
            else
                axs[1+k].set_xlabel("iter ('000)")
            end

            ### Ylabels
            if σlast == true && k == D
                if labels == 0
                    axs[1+k].set_ylabel("σerr")
                else
                    axs[1+k].set_ylabel(labels[k])
                end
            else
                if labels == 0
                    axs[1+k].set_ylabel(string("θ",k))
                else
                    axs[1+k].set_ylabel(labels[k])
                end
            end

            ### Plot true values
            if truevals != 0 && c == C
                axs[1+k].plot([0.0,M]/1e3,truevals[k]*[1.0,1.0],"k:")
            end

        end

    end

    # Display figure if required (on by default)
    if showfig
        display(fig)
    end

    # Return figure
    return fig

end
