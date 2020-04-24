# -*- coding: utf-8 -*-
using Pkg
pkg"add CovarianceMatrices"

using CovarianceMatrices

# +
using LinearAlgebra
using Random
using RCall

include("../src/permute_conditional.jl")
# -

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

permute.conditional = function(y,n,data.complement,estimation){ 
    # uses SCAD/OLS
  permutation = sample(n)

  # SCAD/OLS estimaytion
  fitted = switch(estimation,
                  LM=lm.fit(cbind(1,data.complement),y)$fitted,
                  SCAD=apply(y,2,SCADmod,x=data.complement,
                             lambda=2*sqrt(var(y[,1])*log(ncol(data.complement))/nrow(data.complement))))

  residuals = y - fitted

  result = fitted + residuals[permutation,]
  return(result)
}

set.seed(1111)

n = 100
Z = matrix(rnorm(5*n),n,5) # data complement
X = matrix(rnorm(3*n),n,3)
Y = cbind(X%*%matrix(c(1,1,1),3,1)+rnorm(n)*.1,Z+rnorm(5*n)*.1)

M = cbind(Y,X,Z)

res = permute.conditional(Y,n,Z,"SCAD")

M1 = cbind(res,X,Z)

par(mfrow = c(1,2),mar = c(3,3,3,3))
image.plot(solve(cov(M)))
image.plot(solve(cov(M1)))
"""

M = rcopy(M)

M1 = rcopy(M1)

cov(M)

rcopy(R"cov(M)")

# - Julia `cov` : Compute the covariance matrix of the matrix X along the dimension dims. If corrected is true (the default) then the sum is scaled with n-1, whereas the sum is scaled with n if corrected is false where n = size(X, dims).
# - R `cov` : 

rng = MersenneTwister(1111)

n = 100
Z = randn(rng, (n,5)) # data complement
X = randn(rng, (n,3))
Y = hcat( X .* ones(3)' .+ randn(n), Z .+ 0.1 * randn(rng,(n,5)))

@show size(Y)

M = hcat(Y,X,Z)

res = permute_conditional(Y, n, Z, "SCAD")

M1 = hcat(res, X, Z)

rcopy(R"cov(M1)")

cov(M1, corrected=false)

?cov

sum(eigen(inv(cov(M1))).values) 

#=
"""

# [1] 31.91363

@show S1

# somme les valeurs propres de l'inverse de la covariance de M
S2 = R"""
sum(eigen(solve(cov(M)))$values) 
"""

# [1] 2346.617

@show S2

S3 = R"""
sum(apply(M,2,var)) # somme des variances  des variables qui composent M
"""
# [1] 24.05086
@show S3

# après la permutation la somme les valeurs propres de l'inverse de la 
# covariance de M1 est du même ordre que somme des variances  des variables qui composent M
# on purrait donc tester si

test_covariance = R"abs( sum(eigen(solve(cov(M1)))$values)-sum(apply(M,2,var)))"


@test rcopy(test_covariance) < 10.0

end
=#
