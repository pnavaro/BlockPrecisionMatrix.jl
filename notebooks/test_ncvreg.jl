# -*- coding: utf-8 -*-
using Pkg
pkg"add https://github.com/pnavaro/NCVREG.jl"

using LinearAlgebra
using Random
using NCVREG
using RCall
using Statistics
using Test

rng = MersenneTwister(1234)

n, p = 50, 5
# prepare data
X = randn(rng, n, p)              # feature matrix
a0 = collect(1:p)                # ground truths

y = X * a0 + 0.1 * randn(n) # generate response

XX = hcat(X, randn(rng, n, p))

@rput XX

@rput y

R"library(ncvreg)"
R"scad <- coef(ncvreg(XX, y, lambda=0.2, penalty='SCAD', eps=.0001))"

@rget scad

println( " R scad = $scad")

λ = [0.2]

scad = NCVREG.coef(SCAD(XX, y, λ))

println( " Julia scad = $scad")

mcp = MCP(X, y, λ)

@test maximum(abs.(beta .- mcp.beta)) < 2e-5

yp = XX * mcp.beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("MCP rmse = $rmse")

lasso = Lasso(X, y, λ)

@test maximum(abs.(beta .- lasso.beta)) < 2e-5

yp = XX * lasso.beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("Lasso rmse = $rmse")
