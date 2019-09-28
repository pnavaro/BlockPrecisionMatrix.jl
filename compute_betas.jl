# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

using Distributions, LinearAlgebra, InvertedIndices

sigma = 2.0 * Matrix(I, 3, 3)

X = transpose(rand( MvNormal(sigma), 10))

include("src/utils.jl")
include("src/lasso.jl")

k = 1
nrows, ncols = size(X)
lambda = 2*sqrt(var(X[:,k])*log(ncols)/nrows)
model = LassoRegression( X[:,Not(k)], X[:,k], 15, lambda, 0.01, 1000)


