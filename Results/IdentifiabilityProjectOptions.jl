
# Modules
using Random
using Identifiability
using PyPlot
using KernelDensity
using LinearAlgebra
using Distributions
using JLD2
using StatsBase

# Plot options
plt.style.use("default")

matplotlib.rc("axes", linewidth=0.5)
matplotlib.rc("axes", axisbelow=true)
matplotlib.rc("axes", labelcolor="#545454")
matplotlib.rc("axes", edgecolor="#aaaaaa")
matplotlib.rc("axes", labelsize=8.0)
matplotlib.rc("axes", grid=true)
matplotlib.rc("xtick", color="#aaaaaa")
matplotlib.rc("ytick", color="#aaaaaa")
matplotlib.rc("xtick.major", width=0.5)
matplotlib.rc("ytick.major", width=0.5)
matplotlib.rc("xtick.major", pad=2.0)
matplotlib.rc("ytick.major", pad=2.0)
matplotlib.rc("grid", linewidth=0.5)
matplotlib.rc("grid", color="#fafafa")
matplotlib.rc("font", size=8.0)
matplotlib.rc("font", family="Helvetica")
matplotlib.rc("pdf", fonttype=42)
matplotlib.rc("savefig",transparent=true)
matplotlib.rc("legend", fontsize=8)
matplotlib.rc("legend", fancybox=false)
matplotlib.rc("legend", edgecolor="#545454")
matplotlib.rc("legend", frameon=false)

FigFull = (6.9,5.2)
FigHalf = (3.4,2.55)
FigQuarter = (1.7,1.275)
FigHalfShare = (3.8,2.8)

# Plot colours
pcols = ["#E24933","#535EB8","#00BA38"]
