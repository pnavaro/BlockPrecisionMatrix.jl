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

# # Algorithm
#
# $$
# \mathbf{x}^{(0)} = (x_1^0,...,x_n^0)
# $$
#
# $$
# \mathbf{x}_i^{(k+1)} = argmin_{\omega} f(x_1^{(k+1)}, ..., x_{i-1}^{(k+1)}, \omega, x_{i+1}^{(k)}, ..., x_n^{(k)})
# $$
#
# $$
# x_i : = x_i - \alpha \frac{\partial f}{\partial x_i}(\mathbf{x})
# $$
#
# $$
# \rho_j = \sum_{i=1}^m x_j^{(i)}  (y^{(i)}  - \sum_{k \neq j}^n \theta_k x_k^{(i)} ) = \sum_{i=1}^m x_j^{(i)}  (y^{(i)}  - \hat y^{(i)}_{pred} + \theta_j x_j^{(i)} )
# $$

# +
#Load the diabetes dataset

using ScikitLearn
@sk_import datasets: load_diabetes

diabetes = load_diabetes()

X = diabetes["data"]
y = diabetes["target"]
diabetes

# +
using Statistics


"""
Soft threshold function used for normalized 
data and lasso regression
"""
function soft_threshold(ρ :: Float64, λ :: Float64)
    if ρ + λ <  0
        return (ρ + λ)
    elseif ρ - λ > 0
        return (ρ - λ)
    else
        return 0
    end
end
# -

"""
Coordinate gradient descent for lasso regression 
for normalized data. 

The intercept parameter allows to specify whether 
or not we regularize ``\\theta_0``
"""    
function coordinate_descent_lasso(theta, X, y; lamda = .01, 
        num_iters=100, intercept = false)
    
    #Initialisation of useful values 
    m, n = size(X)
    X .= X ./ sqrt.(sum(X.^2, dims=1)) #normalizing X in case it was not done before
    y_pred = similar(y)
    #Looping until max number of iterations
    for i in 1:num_iters
        
        #Looping through each coordinate
        for j in 1:n
            
            #Vectorized implementation
            X_j = view(X,:,j)
            y_pred .= X * theta
            rho = X_j' * (y .- y_pred  .+ theta[j] .* X_j)
        
            #Checking intercept parameter
            if intercept  
                if j == 0
                    theta[j] =  first(rho) 
                else
                    theta[j] =  soft_threshold(rho..., lamda)
                end
            end

            if !intercept
                theta[j] =  soft_threshold(rho..., lamda)
            end
        end
    end
            
    return vec(theta)
end

function lasso(X, y)
    # Initialize variables
    m,n = size(X)
    initial_theta = ones(Float64,n)
    theta_lasso = Vector{Float64}[]
    lamda = exp10.(range(0, stop=4, length=300)) ./10 #Range of lambda values
    
    #Run lasso regression for each lambda
    for l in lamda
        theta = coordinate_descent_lasso(initial_theta, X, y, lamda = l, num_iters=100)
        push!(theta_lasso, copy(theta))
    end
    return vcat(theta_lasso'...)
end

using BenchmarkTools
@btime theta_lasso = lasso( X, y)

#Plot results
using Plots
n = length(theta_lasso)
p = plot(figsize = (12,8))
for (i,label) in enumerate(diabetes["feature_names"])
    plot!(p, theta_lasso[:,i], label = label)
end
display(p)



