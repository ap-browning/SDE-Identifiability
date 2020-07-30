#=
#
#   PlotScatterMatrix.jl
#
#   Produce scatter plot matrix to visualise MCMC results
#
#   Alexander P. Browning
#       School of Mathematical Sciences
#       Queensland University of Technology
#       ap.browning@qut.edu.au
#       https://alexbrowning.me
#
=#
@doc """
    PlotScatterMatrix(Θ::Array{Float64,3};...)

Produces a scatter plot matrix to view MCMC results.

Inputs:\n
    `Θ`             - MCMC results
    `fig`           - (optional)                            handle to figure to plot
    `burnin`        - (optional, default: 0)                burn-in period for scatters and kernel density estimates
    `skip`          - (optional, default: 1)                skip for thinning MCMC samples for scatters and kernel density estimates
    `lims`          - (optional)                            matrix of axis limits (or column of maximum, where minimum is zero)
    `π_pdf`         - (optional)                            univariate prior probability density function.
    `truevals`      - (optional)                            true values to plot
    `labels`        - (optional, default: ["θ1","θ2",...])  variable names
    `grid`          - (optional, default: true)             plot grid behind scatter and kernel density plots
    `s_size`        - (optional, default: 2)                size of scatters
    `s_alpha`       - (optional, default: 0.5)              transparency of scatters
    `linewidth`     - (optional, default: 1)                linewidth for traces and kernel density estimates
    `t_skip`        - (optional, default: 1)                skip samples when plotting MCMC traces
    `t_burnin`      - (optional, default: 0)                discard samples as burn-in when plotting MCMC traces
    `k_grid`        - (optional, default: 100)              number of x elements to use when computing kernel density estimates
    `k_dispy`       - (optional, default: false)            display y axis labels on univariate kernel density estimates
    `acf`           - (optional, default: false)            plot auto correlation function (ACF)
    `acfmaxlag`     - (optional, default: 100)              maximum lag when plotting ACF
    `showfig`       - (optional, default: true)             show figure once produced
    `bv_scatter`    - (optional, default: true)             if false, plot bivariate kernel density estimate

Outputs:\n
    `fig`      - handle to figure
"""
function PlotScatterMatrix(Θ::Array{Float64,3};
            fig=0,burnin=0,skip=1,lims=0,π_pdf=0,truevals=0,labels=0,grid=true,
            s_size=2,s_alpha=0.5,linewidth=1,t_skip=1,t_burnin=0,k_grid=100,k_dispy=false,
            acf=false,acfmaxlag=100,showfig=true,bv_scatter=true)

    # Problem size
    C,D,S = size(Θ)

    # Number of subplots
    nrows  = D                              # Number rows in grid
    ncols  = acf ? nrows + 2 : nrows + 1    # Number columns in grid
    offset = acf ? 2 : 1                    # Column in which scatters start

    # Row and column to subplot indices
    SubPlotIndex = (row,col) -> [nrows,ncols,(row - 1) * ncols + col]


    # Process MCMC data

        ## Traces
        Θtrace = Θ[1,:,t_burnin+2:t_skip:end]   # Trace data to plot (inc. burnin and skip)
        iter   = (t_burnin+1:t_skip:S-1) / 1e3  # Trace iteration to plot on x axis (units of '000)

        ## Samples
        indices = burnin+2:skip:S               # Indices of samples to keep for kernel density and scatters
        ns_each = length(indices)               # Number of samples per chain
        samples = zeros(D,C*ns_each)            # Pool samples
        for c = 1:C
            samples[:,ns_each*(c-1)+1:ns_each*c] = Θ[c,:,indices]
        end

    # Process inputs

        ## Labels
        if labels == 0
            labels = Array{String,1}(undef,D)
            for i = 1:D
                labels[i] = string("θ",i)
            end
        end

        ## Create new figure?
        if fig == 0
            fig = figure()
        else
            fig = figure(fig.number)
            clf()
        end

        ## Given axis limits?
        if lims != 0 && ndims(lims) == 1
            lims = [0*lims lims]    # Limits as a matrix, if only upper bound given
        end

    # x- and y-axis grid nice (5 ticks in each x and y)
    function nicegrid(ax)
        ax.set_xticks(range(ax.get_xlim()...,length=5))
        ax.set_yticks(range(ax.get_ylim()...,length=5))
    end

    # Nice x-axis labels (label every second)
    function nice_xlabs(ax)
        vals = ax.get_xticks()
        labs = string.(vals)
        labs[2:2:end] .= ""
        ax.set_xticklabels(labs)
    end

    # Nice y-axis labels (label every second)
    function nice_ylabs(ax)
        vals = ax.get_yticks()
        labs = string.(vals)
        labs[2:2:end] .= ""
        ax.set_yticklabels(labs)
    end

################################################################################

    # Plot traces (first column)
    for i = 1:D

        ## Select subplot
        ax = subplot(SubPlotIndex(i,1)...)

        ## Plot trace from first chain
        ax.plot(iter,Θtrace[i,:],linewidth=linewidth,color="#535EB8")

        ## Nice axis limits
        ax.set_xlim([0.0,S-1]/1e3)  # x-lims
        if lims != 0                # y-lims (if given for the variable)
            ax.set_ylim(lims[i,:])
        end

        ## Grid and axis ticks
        nicegrid(ax)
        ax.grid("on")

        ## Hide xticklabels if not last row
        if i != D
            ax.set_xticklabels([])
        else
            nice_xlabs(ax)          # Number every second x-tick
        end
        nice_ylabs(ax)              # Number every second y-tick

        ## Axis labels
        if i == D                   # x-label (if last row)
            ax.set_xlabel("Iteration ('000)")
        end
        ax.set_ylabel(labels[i])    # y-label

        ## Plot "true" values
        if truevals != 0
            ax.plot([0.0,S-1],truevals[i]*[1.0,1.0],"k:")
        end

    end # (for i = 1:D)

    # Plot acf (if required)
    if acf
        for i = 1:D

            # Compute acf
            l = 0:acfmaxlag
            ρ = autocor(Θ[1,i,(burnin+1):end],l)

            # Select subplot
            ax = subplot(SubPlotIndex(i,2)...)

            # Plot ACF
            ax.plot(l,ρ,linewidth=linewidth,color="#535EB8")

            # Plot zero line ("target")
            ax.plot([0.0,acfmaxlag],[0.0,0.0],"k:")

            # Axis limits
            ax.set_xlim([0.0,acfmaxlag])    # set x limits
            ax.set_ylim([-0.5,1])           # set y limits

            # Hide xticklabels if not last row
            if i != d
                ax.set_xticklabels([])
            else
                ax.set_xlabel("lag")
            end

            # Show grid
            ax.grid("on")

        end # (for i = 1:D)
    end # (if acf)

    # Plot scatters
    for i = 1:D-1       # Loop through cols (x axis)
        for j = i+1:D   # Loop through rows (y axis)

            # Select subplot
            ax = subplot(SubPlotIndex(j,i+offset)...)

            # Plot bivariate scatters
            if bv_scatter || lims == 0

                # Plot scatters
                ax.scatter(samples[i,:],samples[j,:],s_size,color="#535EB8",alpha=s_alpha)

                # Set limits if given
                if lims != 0
                    ax.set_xlim(lims[i,:])
                    ax.set_ylim(lims[j,:])
                end

                # Grid and axis ticks
                ax.grid("on")
                nicegrid(ax)

                # Axis labels
                if j != D
                    ax.set_xticklabels([])  # Hide if not last row
                else
                    nice_xlabs(ax)          # Nice x-ticks
                end
                ax.set_yticklabels([])      # y-label always zero

                # Plot "true" values
                if truevals != 0
                    ax.scatter([truevals[i]],[truevals[j]],2*s_size,marker="v",color="k")
                end

            # Plot bivariate kernel density
            else

                # Compute KDE from samples
                B  = kde(samples[[i,j],:]')

                # Plotting range (lims must be given here)
                p1 = range(lims[i,1],lims[i,2],length=200)
                p2 = range(lims[j,1],lims[j,2],length=200)

                # Calculate kernel density estimate of posterior
                f  = pdf(B,p1,p2)

                # Plot contour of kernel density estimate
                ax.contourf(p1,p2,f',extend="min")

                # Axis limits and nice ticks
                ax.set_xlim(lims[i,:])
                ax.set_ylim(lims[j,:])
                nicegrid(ax)

                # Axis labels
                if j != D
                    ax.set_xticklabels([])  # Hide if not last row
                else
                    nice_xlabs(ax)          # Nice x-ticks
                end
                ax.set_yticklabels([])      # y-label always zero

                # Plot "true" values
                if truevals != 0
                    ax.scatter([truevals[i]],[truevals[j]],2*s_size,marker="v",color="w")
                end

            end

        end # (j = i+1:D)

    end # (i = 1:D-1)

    # Plot univariate kernel density estimates (lims must be given)
    if lims != 0 && k_grid != 0
        for i = 1:D

            # Select subplot
            ax = subplot(SubPlotIndex(i,i+offset)...)

            # x grid to plot
            x = range(lims[i,1],lims[i,2],length=k_grid)

            # Plot prior, if given (assume independent priors)
            if π_pdf != 0

                prior = zeros(size(x))
                for j = 1:length(prior)
                    θx   = 0.5*sum(lims,dims=2)[:]; θx[i] = x[j];
                    prior[j] = π_pdf(θx)[i]
                end

                ax.fill_between(x,0,prior,alpha=.3,color="#E24933")
                ax.plot(x,prior,linewidth=linewidth,color="#E24933")

            end

            # Calculate kernel density estimate
            B = kde(samples[i,:])
            f  = pdf(B,x)

            # Plot kernel density estimate of posterior
            ax.fill_between(x,0,f,alpha=.3,color="#535EB8")
            ax.plot(x,f,linewidth=linewidth,color="#535EB8")

            # Axis limits and nice ticks
            ax.set_xlim(lims[i,:])      # Set x limits
            ylims = ax.get_ylim()       # Current y limits
            ax.set_ylim([0,ylims[2]])        # Set lower y bound to zero
            nicegrid(ax)
            ax.grid("on")

            # Display y labels if required
            if !k_dispy
                ax.set_yticklabels([])
            end

            # Display x labels only if last row
            if i != D
                ax.set_xticklabels([])
            else
                nice_xlabs(ax)
            end

            # Display "true" values
            if truevals != 0
                ylims = ax.get_ylim()
                ax.plot(truevals[i]*[1.0,1.0],[ylims[1],ylims[2]],"k:")
            end

        end # (for i = 1:D)

    end # (if lims != 0)

    # x-labels along last row
    for i = 1:D

        # Select subplot
        ax = subplot(SubPlotIndex(nrows,i+offset)...)

        # x axis label
        ax.set_xlabel(labels[i])

    end

    # Display figure if required (on by default)
    if showfig
        display(fig)
    end

    # Return figure
    return fig

end
