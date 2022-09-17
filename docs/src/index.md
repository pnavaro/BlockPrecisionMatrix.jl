# BlockPrecisionMatrix.jl

[A Visual Explanation of Statistical Testing](https://www.jwilber.me/permutationtest/)


In the animation above, they illustrate a test on the mean. In our case, we want to apply the ideas of permutation tests (inverse of the covariance matrix) but to identify the null blocks of a precision matrix. A block will be null in the following situation.

Consider the blocks associated with the variables ``X`` and ``Y`` of a data set D such that

``D = (X,Y,Z)`` with ``X \in R^{n,p_X}, Y \in R^{n,p_Y}, Z \in R^{n,p_Z}``

```math
\begin{aligned}
X & = \beta_X [1,Z] + \epsilon_X \\
Y & = \beta_Y [1,Z] + \epsilon_Y
\end{aligned}
```

then the block ``P_{XY}`` is null if ``\epsilon_X`` and ``\epsilon_Y`` are independent ie if the correlation matrix of ``\epsilon_X`` with ``\epsilon_Y`` is null.
 
So we apply the permutations to the residues ``\epsilon_X`` and/or
``epsilon_Y``. To do this, we first need to estimate the
linear model (this is where we use the SCAD estimation), recover the residuals,
permute them and then reconstruct the
data corresponding to the permuted residuals with [`permute_scad`](@ref).

To finish the test, we need to estimate the precision matrices associated with the permutations.
We use the callable type [`StatTest`](@ref).

```@docs
permute_scad
```

```@docs
StatTest
```

```@docs
BlockPrecisionMatrix.cov_simu
```
```@docs
BlockPrecisionMatrix.structure_cov
```
```@docs
BlockPrecisionMatrix.generate_data
```
```@docs
BlockPrecisionMatrix.permute
```
```@docs
BlockPrecisionMatrix.iwt_block_precision
```
