# -*- coding: utf-8 -*-
using Pkg
pkg"add https://github.com/pnavaro/NCVREG.jl"

using LinearAlgebra
using NCVREG
using Plots
using Random
using RCall
using Statistics
using PrecisionMatrix


# # Code R (Madison)

R"""
library(spam)
library(ncvreg)
library(fields)
set.seed(1111)

SCADmod = function(yvector,x,lambda){
  regSCAD = ncvreg::ncvreg(x,yvector,
                           penalty='SCAD',lambda=lambda)
  fitted = cbind(1,x) %*% regSCAD$beta
  return(fitted)
}

set.seed(1111)

n = 100
Z = matrix(rnorm(5*n),n,5) # data complement
X = matrix(rnorm(3*n),n,3)
Y = cbind(X%*%matrix(c(1,1,1),3,1)+rnorm(n)*.1,Z+rnorm(5*n)*.1)

M = cbind(Y,X,Z)

SCADmod = function(yvector,x,lambda){
  regSCAD = ncvreg::ncvreg(x,yvector,
                           penalty='SCAD',lambda=lambda)
  return(cbind(1,x) %*% regSCAD$beta)
}

permutation = sample(n)

fitted = apply(Y,2,SCADmod,x=Z,lambda=2*sqrt(var(Y[,1])*log(ncol(Z))/nrow(Z)))

residuals = Y - fitted

res = fitted + residuals[permutation,]
print(dim(res))
M1 = cbind(res,X,Z)

par(mfrow = c(1,2), mar=c(0,0,0,0))
image.plot(solve(cov(M)),  asp=1, axes=F, legend.shrink=.4 )
image.plot(solve(cov(M1)),  asp=1, axes=F, legend.shrink=.4)
"""

# +
@rget permutation
@rget X
@rget Y
@rget Z

function permutation_conditional(permutation, Y, Z)
    row_y, col_y = size(Y)
    row_z, col_z = size(Z)
    
    @assert row_y == row_z
    
    fitted = similar(Y)
    for j in 1:col_y
        y = Y[:,j]
        λ = 2*sqrt(var(Y[:,1])*log(col_z)/row_z)
        beta = NCVREG.coef(SCAD(Z, y, [λ]))  
        fitted[:,j] .= vec(hcat(ones(row_z),Z) * beta)
    end
    
    residuals = Y .- fitted
    
    #permutation = randperm(rng, row_y)
        
    return fitted .+ residuals[permutation,:]
end

M  = hcat(Y, X, Z)
result_jl = permutation_conditional(permutation, Y, Z)
M1 = hcat(result_jl, X, Z)

M1 ≈ rcopy(M1)
# -

image = plot(layout=(1,2))
heatmap!(image[1,1], inv(cov(M)), aspect_ratio=:equal, c=cgrad([:white, :black]))
heatmap!(image[1,2], inv(cov(M1)), aspect_ratio=:equal, c=cgrad([:white,:black ]))

# # Intialize data with Julia

rng = MersenneTwister(1111)
n = 100
Z = randn(rng, (n,5)) # data complement
X = randn(rng, (n,3))
Y = hcat( X * ones(3) .+ 0.1 .* randn(n), Z .+ 0.1 * randn(rng,(n,5)))
M = hcat(Y,X,Z)
permutation = randperm(n)
result_jl = permutation_conditional(permutation, Y, Z)
M1 = hcat(result_jl, X, Z)
image = plot(layout=(1,2), axis=false, grid=false)
heatmap!(image[1,1], inv(cov(M)), aspect_ratio=:equal, c=cgrad([:white, :black]), levels=4)
heatmap!(image[1,2], inv(cov(M1)), aspect_ratio=:equal, c=cgrad([:white,:black ]), levels=4)



