# BlockPrecisionMatrix.jl

This computation is a nice exemple to test parallelization in Julia. In notebooks directory
you find different ways to parallelize the same process. I used `pmap`, `Distributed channel` and `threads`.

![CI](https://github.com/pnavaro/BlockPrecisionMatrix.jl/workflows/CI/badge.svg)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://pnavaro.github.io/BlockPrecisionMatrix/dev)

```bash
git clone https://github.com/pnavaro/BlockPrecisionMatrix.jl.git
cd BlockPrecisionMatrix
julia --project
```

# Some linked packages 

- [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl)
- [Lasso.jl](https://github.com/JuliaStats/Lasso.jl)
- [QuantileRegression.jl](https://github.com/pkofod/QuantileRegression.jl)
- [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl)

# References

Xia, L. , Huang, X. , Wang, G. and Wu, T. (2017) Positive-Definite Sparse Precision Matrix Estimation. Advances in Pure Mathematics, 7, 21-30. doi: 10.4236/apm.2017.71002.
