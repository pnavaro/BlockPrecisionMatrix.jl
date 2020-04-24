# -*- coding: utf-8 -*-
using Pkg
pkg"add https://github.com/pnavaro/NCVREG.jl"

using LinearAlgebra
using Random
using RCall
using Plots
using NCVREG


# somme les valeurs propres de l'inverse de la covariance de M1

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

fitted + residuals[permutation,]
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

row_y, col_y = size(Y)
row_z, col_z = size(Z)
fitted = zero(Y)
for j in 1:col_y
    y = Y[:,j]
    λ = 2*sqrt(var(Y[:,1])*log(col_z)/row_z)
    beta = NCVREG.coef(SCAD(Z, y, [λ]))  
    fitted[:,j] .= vec(hcat(ones(row_z),Z) * beta)
end

residuals = Y .- fitted
    
result_jl = fitted .+ residuals[permutation,:]
M1 = hcat(result_jl, X, Z)

M1 ≈ rcopy(M1)
# -

image = plot(layout=(1,2))
heatmap!(image[1,1], inv(cov(M)), aspect_ratio=:equal, c=cgrad([:blue, :white,:red, :yellow]))
heatmap!(image[1,2], inv(cov(M1)), aspect_ratio=:equal, c=cgrad([:blue, :white,:red, :yellow]))

# # Intialize data with Julia

# +
rng = MersenneTwister(1111)

n = 100
Z = randn(rng, (n,5)) # data complement
X = randn(rng, (n,3))
Y = hcat( X * ones(3) .+ 0.1 .* randn(n), Z .+ 0.1 * randn(rng,(n,5)))
M = hcat(Y,X,Z)
