# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

using GLM, LinearAlgebra, Plots, Random, Statistics

# # Prepare data

# +
rng = MersenneTwister(0)
n, p = 10, 1
X = randn(rng, (n, p))
b = collect(1:p) 
y = 0.1*randn(rng, n) .+  X * b

xmin, xmax = extrema(X[:,1])
npoints = 100
x = LinRange(xmin, xmax, npoints)
# -

# # Linear regression with obvious version

XX = hcat(ones(n), X)
beta = inv(XX'XX) * XX'y
scatter( X[:,1], y)
plot!(x, hcat(ones(npoints),x) * beta)

# # Version with QR factorisation
# The `\` operator is the short-hand for
# ```julia
# Q, R = qr(X)
# β = inv(factorize(R)) * Q.'y
# ```

XX = hcat(ones(n), X)
beta = XX \ y

plot!(x, hcat(ones(npoints),x) * beta)

# # Version with singular values decomposition

# ```jl
# β = pinv(X) * y
# ```
#
# means
#
# ```jl
# U, S, V = svd(X)
# β = V * diagm(1 ./ S) * U.' * y
# ```

XX = hcat(ones(n), X)
beta = pinv(XX) * y
scatter( X[:,1], y)
plot!(x, hcat(ones(npoints),x) * beta)
# ## With GLM.jl

U, S, V = svd(XX)
β = V * diagm(1 ./ S) * U' * y

# +
using GLM

fitted = lm(XX, y)
# -

scatter( X[:,1], y)
plot!(x, GLM.predict(fitted, hcat(ones(npoints),x)))
