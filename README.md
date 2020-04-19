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
julia> pkg"instantiate"
julia> pkg"add https://github.com/pnavaro/NCVREG.jl.git"
julia> using IJulia
julia> notebook(dir=pwd())
[ Info: running ...
```

# Some linked packages 

- [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl)
- [Lasso.jl](https://github.com/JuliaStats/Lasso.jl)
- [QuantileRegression.jl](https://github.com/pkofod/QuantileRegression.jl)
- [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl)
