
using GLMNet
using InvertedIndices
using LinearAlgebra
using PrecisionMatrix
using RCall
using Test

R"library(glmnet)"


using RCall, Random, GLMNet, InvertedIndices
using Statistics, Distributions

using LinearAlgebra
nrows, ncols = 8, 5
@rput ncols
@rput nrows
R""" 
X0 <- matrix(1:nrows*ncols, nrows, ncols)
Sigmainv <- .25^abs(outer(1:ncols,1:ncols,"-"))
X <- backsolve( chol(Sigmainv), X0)"""

X = rcopy(R"X")

result = R"""
PrecXia = function(X){
  n = nrow(X)
  betahat = matrix(0,ncol(X)-1,ncol(X))
  reshat  = X
  for (k in 1:ncol(X)){
    fitreg  = glmnet::glmnet(X[,-k],X[,k],family="gaussian",
        lambda = 2*sqrt(var(X[,k])*log(ncol(X))/nrow(X)),
                             standardize = FALSE) 
    betahat[,k] = as.vector(fitreg$beta)
    reshat[,k]  = X[,k]-as.vector(predict(fitreg,X[,-k]))
  }
  rtilde = cov(reshat)*(n-1)/n
  rhat   = rtilde
  for (i in 1:(ncol(X)-1)){
    for (j in (i+1):ncol(X)){
      rhat[i,j] = -(rtilde[i,j]+rtilde[i,i]*betahat[i,j]+rtilde[j,j]*betahat[j-1,i])
      rhat[j,i] = rhat[i,j]
    }
  }
  Tprec  = 1/rhat
  TprecStd = Tprec
  for (i in 1:(ncol(X)-1)){
    for (j in (i+1):ncol(X)){
      Tprec[i,j] = Tprec[j,i] = rhat[i,j]/(rhat[i,i]*rhat[j,j])
      thetahatij = (1+(betahat[i,j]^2*rhat[i,i]/rhat[j,j]))/(n*rhat[i,i]*rhat[j,j])   
      TprecStd[i,j] = TprecStd[j,i] = Tprec[i,j]/sqrt(thetahatij)
    }
  }
  
  MatPrecXia = list(Tprec=Tprec,TprecStd=TprecStd)
}

PrecXia(X)
"""

@testset "Prec Xia" begin

Tprec_jl, TprecStd_jl = PrecisionMatrix.prec_xia(X)

result_R = rcopy(result)

@test Tprec_jl ≈ result_R[:Tprec]
@test TprecStd_jl ≈ result_R[:TprecStd]

end
