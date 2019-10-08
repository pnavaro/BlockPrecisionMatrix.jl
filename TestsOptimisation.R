rm(list=ls())

library(mvtnorm)
library(fields)
library(parallel)
library(profvis)
library(here)

source(here('R','FonctionsSimu.R'))
#source('~/Dropbox/code_BlockCovarianceTest/blocks_loop.R', echo = TRUE)
source(here('R','Precision_IWT_function.R'), echo=TRUE)
source(here('R','utilities.R'), echo = TRUE)

p        = 20 # c(20, 100)
n        = 500 # c(100, 200, 500, 1000)
b        = 3 # c(3, 5, 5, 20)
blocsOn  = list(c(1,3))

# simulation des donnees
resBlocs = StructureCov(p, b, blocsOn, seed = 2)
resBlocs$blocs
D        = runif(p, 10^-4, 10^-2)
resmat   = CovSimu(resBlocs$blocs, resBlocs$indblocs, blocsOn, D = D)
image.plot(resmat$CovMat)
image.plot(resmat$PreMat)

datas     = rmvnorm(n, rep(0, p), sigma = resmat$CovMat)
p.part     = sapply(1:length(resBlocs$indblocs), function(i) length(resBlocs$indblocs[[i]]))
(blocks     = rep(1:b, p.part))

library(microbenchmark)
X = as.matrix(datas[,-1])
Y = datas[,1]
res <- microbenchmark(glmnet(X, Y, family = "gaussian", lambda = 2*sqrt(var(Y)*log(ncol(X))/nrow(X))))

print(res)





# test de la procedure
p=profvis({
  IWT_Block = IWT_Block_precision(datas, blocks)
})

htmlwidgets::saveWidget(p, "profile.html")
browseURL("profile.html")

# comparaison glmnet et penalized
X = as.matrix(datas[,-1])
Y = datas[,1]
library(rbenchmark)
benchmark(glmnet(X, Y, family = "gaussian", lambda = 2*sqrt(var(Y)*log(ncol(X))/nrow(X)), standardize = FALSE),
          penalized(Y, X, lambda1 = 2*sqrt(var(Y)*log(ncol(X))/nrow(X))))

#### optimisation PrecXia

X = datas

benchmark({Xc  = scale(X,center=TRUE,scale=FALSE)
betahat = sapply(1:ncol(X),FUN=function(k) as.vector(glmnet(X[,-k],X[,k],family="gaussian",lambda = 2*sqrt(var(X[,k])*log(ncol(X))/nrow(X)),standardize = FALSE)$beta))
reshat2 = sapply(1:ncol(X),FUN=function(k) Xc[,k] - colSums(t(Xc[,-k])*betahat[,k])) 
},{  Xc      = scale(X,center=TRUE,scale=FALSE)
betahat = sapply(1:ncol(X),FUN=function(k) as.vector(glmnet(X[,-k],X[,k],family="gaussian",lambda = (1:10)/5*sqrt(var(X[,k])*log(ncol(X))/nrow(X)),standardize = FALSE)$beta))
reshat  = sapply(1:ncol(X),FUN=function(k) Xc[,k] - colSums(t(Xc[,-k])*betahat[,k])) 
})


PrecXia = function(X){
  n = nrow(X)
  
  Xc      = scale(X,center=TRUE,scale=FALSE)
  betahat = sapply(1:ncol(X),FUN=function(k) as.vector(glmnet(X[,-k],X[,k],family="gaussian",lambda = (1:20)/10*sqrt(var(X[,k])*log(ncol(X))/nrow(X)),standardize = FALSE)$beta))
  reshat  = sapply(1:ncol(X),FUN=function(k) Xc[,k] - colSums(t(Xc[,-k])*betahat[,k])) 

  rtilde = cov(reshat)*(n-1)/n
  rhat   = rtilde
  for (i in 1:(ncol(X)-1)){
    for (j in (i+1):ncol(X)){
      rhat[i,j] = -(rtilde[i,j]+rtilde[i,i]*betahat[i,j]+rtilde[j,j]*betahat[j-1,i])
      rhat[j,i] = rhat[i,j]
    }
  }
  TprecStd = 1/rhat
  for (i in 1:(ncol(X)-1)){
    for (j in (i+1):ncol(X)){
      thetahatij = (1+(betahat[i,j]^2*rhat[i,i]/rhat[j,j]))/(n*rhat[i,i]*rhat[j,j])   
      TprecStd[i,j] = TprecStd[j,i] = rhat[i,j]/((rhat[i,i]*rhat[j,j])*sqrt(thetahatij))
    }
  }
  
  return(TprecStd)
}

benchmark(PrecXia(X))
