# PrecisionMatrix


## Run the R code

  - Install packages dependencies

```bash
Rscript install.R
Rscript  TestsOptimisation.R
```

## Run the Julia code

```bash
git clone https://github.com/MarieMorvan/PrecisionMatrix.git
cd PrecisionMatrix
julia --project
```

```julia
julia> using Pkg
julia> Pkg.instantiate()
julia> using IJulia
julia> notebook(dir=pwd())
[ Info: running ...
```
