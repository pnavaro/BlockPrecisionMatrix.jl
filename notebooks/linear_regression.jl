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

0.1 * randn(10)

# +
using Random, Plots

n = 20
X = rand(n,1)
y = 1 .+ 2 .* X[:,1] + 0.2 .* randn(n)
XX = hcat(ones(n), X)
beta = inv(XX'XX) * XX'y
scatter( X[:,1], y)
x = LinRange(0,1,100)
plot!(x, hcat(ones(100),x) * beta)
# -



pl

# +

n, p = size(x)
vec(model.beta * vcat(ones(n), x)')
# -





using GLM, RDatasets

form = dataset("datasets", "Formaldehyde")

lm1 = fit(LinearModel, @formula(OptDen ~ Carb), form)

lm1.model

predict(lm1, Table(LinRange(0.1,0.9,100)))

Table

typeof(form.Carb)

using Random
blocks = shuffle(4:8)
index_y = collect(1:10)

blocks in index_y

prostate = dataset("datasets", "Prostate")








