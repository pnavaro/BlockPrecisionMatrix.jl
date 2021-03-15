# PrecisionMatrix.jl

This computation is a nice exemple to test parallelization in Julia. In notebooks directory
you find different ways to parallelize the same process. I used `pmap`, `Distributed channel` and `threads`.

![CI](https://github.com/pnavaro/PrecisionMatrix/workflows/CI/badge.svg)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://pnavaro.github.io/PrecisionMatrix/dev)

```bash
git clone https://github.com/pnavaro/PrecisionMatrix.git
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
