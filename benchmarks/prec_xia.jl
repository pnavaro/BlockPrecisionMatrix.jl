using BenchmarkTools
using Distributions
using GLMNet
using InvertedIndices
using LinearAlgebra
using Random
using RCall
using Statistics
using Test
using TimerOutputs

reset_timer!()

R"library(glmnet)"

nrows, ncols = 500, 20
@rput ncols
@rput nrows
R""" 
X0 <- matrix(1:nrows*ncols, nrows, ncols)
Sigmainv <- .25^abs(outer(1:ncols,1:ncols,"-"))
X <- backsolve( chol(Sigmainv), X0)"""

X = rcopy(R"X")

R"""
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
"""

function prec_xia!( Tprec, TprecStd, X :: Array{Float64, 2} )
    
    n, p = size(X)

    betahat = zeros(Float64,(p-1,p))
    reshat  = copy(X)
    y = zeros(Float64, n)
    
    @inbounds for k in 1:p
        x = view(X, :, Not(k))
        y .= X[:, k]
        λ = [2*sqrt(var(y)*log(p)/n)]
        fitreg = glmnet(x, y, lambda = λ, standardize=false)
        betahat[:,k] .= vec(fitreg.betas)
        y .= vec(GLMNet.predict(fitreg,x))
        reshat[:,k]  .= view(X, :, k) .- y
    end
    
    rtilde = cov(reshat) .* (n-1) ./ n
    rhat   = rtilde
    
    @inbounds for i in 1:p-1
        for j in (i+1):p
            rhat[i,j] = -(rtilde[i,j]+rtilde[i,i]*betahat[i,j]+rtilde[j,j]*betahat[j-1,i])
            rhat[j,i] = rhat[i,j]
        end
    end
    
    @inbounds for i in eachindex(Tprec, rhat)
        Tprec[i] = 1 ./ rhat[i]
    end

    TprecStd .= Tprec

    for i in 1:(p-1)
        for j in (i+1):p
            Tprec[i,j] = Tprec[j,i] = rhat[i,j]/(rhat[i,i]*rhat[j,j])
            thetahatij = (1+(betahat[i,j]^2  * rhat[i,i]/rhat[j,j]))/(n*rhat[i,i]*rhat[j,j])   
            TprecStd[i,j] = TprecStd[j,i] = Tprec[i,j]/sqrt(thetahatij)
        end
    end

end

@timeit "R" result = R"PrecXia(X)"

Tprec_jl = zeros(ncols, ncols)
TprecStd_jl = zeros(ncols, ncols)
prec_xia!(Tprec_jl, TprecStd_jl, X)

@timeit "Julia" prec_xia!(Tprec_jl, TprecStd_jl, X)

result_R = rcopy(result)

@test Tprec_jl ≈ result_R[:Tprec]
@test TprecStd_jl ≈ result_R[:TprecStd]

print_timer()

println()

@btime prec_xia!(Tprec_jl, TprecStd_jl, X)
