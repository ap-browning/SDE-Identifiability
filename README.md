# Identifiability

 Code to perform identifiability analysis for stochastic differential equations. Repository is supplementary material for the preprint "Listen to the noise: identifiability analysis for stochastic differential equation models in systems biology" available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.08.10.245233v1).

 The majority of the code concerts the `Julia` module `Identifiability` that performs practical identifiability analysis using pseudo-marginal Markov-chain Monte-Carlo (PM-MCMC). This repository also contains scripts to perform structural identifiability analysis using the moment equations in `DAISY` (Bellu 2007), a package written for the `REDUCE` computer algebra system.

## Practical identifiability analysis in `Julia`

### Getting started

Ensure `Julia` is installed (see *Required software*) and download the repository, in its entirety, to your machine. You should then run `Install_Required_Packages.jl` from the `Module` folder.

Use the following commands to add the module to your current search path, and load the module:
```
  push!(LOAD_PATH,"/path/to/module/folder/")  # Add to load path
  using Identifiability                       # Load module
```
If using Windows, ensure to escape the backslashes in the path: `C:\\path\\to\\module\\folder`, or use Unix style forward slashes.


### Module
The module `Identifiability` provides access to the following functions, each thoroughly documented.
  - `SimulateSDE()` to generate synthetic data and/or simulate data from the SDE
  - `SimulateSSA()` to generate synthetic data and/or simulate data from the SSA
  - `MetropolisHastings()` to perform MCMC with the MH algorithm (log-likelihood provided)
  - `DeterministicMH()` to perform MCMC for an ODE model
  - `PseudoMarginalMH()` perform PM-MCMC with the MH algorithm
  - `GetLogPrior` (MCMC setup)
  - `OptimalProposal()` (MCMC setup)
  - `PriorPDF()` to plot prior distribution
  - `Diagnostics()` to calculate R̂ and *n*<sub>eff</sub> diagnostics
  - `CredibleIntervals()` to estimate posterior credible intervals
  - `PosteriorPredictiveQuantiles()` to estimate quantiles of the posterior predictive distribution

The following plotting functions are also available:
  - `PlotScatterMatrix()` to plot MCMC results in a scatter plot matrix
  - `PlotTraces()` to plot only MCMC traces (i.e., from a pilot run)

All functions are thoroughly documented. To obtain documentation for each function, type (for the `PseudoMarginalMH()` function)
```
  ?PseudoMarginalMH()
```

### Results

All results in the main and supporting material documents can be obtained by running the corresponding script in the `Results` folder. By default `load = true` the results are loaded from a `.jld2` file rather than recomputed. Before running scripts with `load = true` ensure the working directory of the `Julia` session, `pwd()` is set to the results folder:
```
  cd("/path/to/results/folder")
```
Approximate runtimes for the full computation of the results for each model (using a 3.7GHz Quad-Code i7 desktop running Windows 10), and figures produced, are given below.

| Script              | Figure(s)       | Runtime     |
|---------------------|-----------------|-------------|
| `Figure3.jl`        | Figure 3        | 3 seconds   |
| `M1_BirthDeath.jl`  | Figures 4 & 5   | 2 hours     |
| `M2_TwoPool.jl`     | Figures 6 – 8   | 17 hours    |
| `M3_SEIR.jl`        | Figures 9 & 10  | 7 hours 	  |
| `M4_BetaIG.jl`  	  | Figure 11 – 13  | 55 hours    |

Note that the code uses the `.Threads` module to run four MCMC chains simultaneously on CPU threads. Use the `nthreads()` command to verify the number of threads in the `JULIA_NUM_THREADS` environment variable. For more information on setting the number of threads in Julia visit [julialang.org](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1).

## Structural identifiability analysis in `DAISY`

Input files to perform structural identifiability analysis in `DAISY` are provided in the `DAISY` folder. Output files (`_Result.txt`) are also provided.

Once `DAISY` and `REDUCE` (these instructions are for `redpsl`, command line `REDUCE`) are installed (see *Required software*) are installed, run `redpsl` / `REDUCE` and load `DAISY` by typing `daisy()$` press enter. You must first tell `DAISY` to output a file using the `OUT` command, before inputting the file with the `IN` command. For example, to run `DAISY` on the `BirthDeathO1.txt` (perform identifiability analysis on the birth-death ODE model), outputting results to `BirthDeathO1_Result.txt`, use the following commands:
```
OUT "/path/to/daisy/folder/M1_BirthDeath/BirthDeathO1_Result.txt"$
IN "/path/to/daisy/folder/M1_BirthDeath/BirthDeathO1.txt"$
SHUT "/path/to/daisy/folder/M1_BirthDeath/BirthDeathO1_Result.txt"$
```
Run `CLEAR ALL$` before calling `DAISY` again.

Note that `DAISY` can take a significant amount of time to run, depending on the complexity of the model. The runtimes for each model are provided below:

| Model               | Script                 | Runtime   |
|---------------------|------------------------|-----------|
| Birth Death         | `_ODE.txt`             | <1 s      |
|                     | `_SDE.txt`             | <1 s      |
| Two Pool            | `_O1.txt`              | <1 s      |
|                     | `_O2.txt`              | <1 s      |
| Epidemic            | `_ODE.txt`             | 5 s       |
|                     | `_SDE_MeanField.txt`   | 1 m       |
|                     | `_SDE_PairWise.txt`    | 16 h      |
|                     | `_SDE_Gaussian.txt`    | 7 h       |

All `DAISY` input files are well commented, and correspond to moment equations derived in the main document and the supporting material document.

## Required software

  - `Julia` can be downloaded from [julialang.org](https://julialang.org/downloads/) or on macOS using `homebrew`: just run `brew cask install julia` in terminal.
  - All `Julia` packages used are available from the standard package installed. Run `Module/Install_Required_Packages.jl` in `Julia` to ensure all required packages are installed.
  - `DAISY`, along with instructions for installing `REDUCE` and tutorials for using `DAISY`, are available from [https://daisy.dei.unipd.it](https://daisy.dei.unipd.it)

 *(Recommended) I recommend the [Juno IDE](https://junolab.org) for `Julia` available for [Atom](https://atom.io)*


## References
1. Bellu G, Saccomani MP, Audoly S, D'Angiò L. 2007 DAISY: A new software tool to test global identiability of biological and physiological systems. *Comput. Meth. Prog. Bio.* **88**, 52-61.
