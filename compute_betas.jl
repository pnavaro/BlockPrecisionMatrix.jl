# -*- coding: utf-8 -*-
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
using GLMNet, Plots
using BenchmarkTools

# +
using ScikitLearn
@sk_import datasets: load_diabetes

diabetes = load_diabetes()

X = diabetes["data"]
y = diabetes["target"];
# -

"""
Soft threshold function used for normalized 
data and lasso regression
"""
function soft_threshold(rho :: Float64, lamda :: Float64)
    if rho < - lamda
        return (rho + lamda)
    elseif rho >  lamda
        return (rho - lamda)
    else
        return 0
    end
end


# +
"""
Coordinate gradient descent for lasso regression.
"""    
function compute_betas(X, y, λ; num_iters=100)
    
    m, n = size(X)
    β = ones(Float64, n)
    mse_next = mean(0.5 * (y .- X * β).^2 .+ λ * norm(β))
    mse_last = mse_next
    training_errors = Float64[]
    i = 1
    #Looping until max number of iterations
    while ( mse_next <= mse_last && i < num_iters )
        #Looping through each coordinate
        for j in 1:n
            
            #Vectorized implementation
            X_j = view( X, :, j)
            y_pred = X * β
            rho = X_j' * (y .- y_pred  .+ β[j] .* X_j)
        
            β[j] = soft_threshold(rho..., λ)
        end
        
        mse_last = mse_next
        mse_next = mean(0.5 * (y .- X * β).^2 .+ λ * norm(β))
        push!(training_errors, mse_last)
        i += 1
        
    end
        
    β, training_errors
            
end
# -

λ = 0.01
@btime β, errors = compute_betas(X, y, λ; num_iters=71)

β

# +
k = 1
nrows, ncols = size(X)

model = glmnet( X, y, lambda = [0.1], standardize=false)
# -

model.betas


