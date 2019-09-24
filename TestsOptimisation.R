rm(list=ls())

library(mvtnorm)
library(fields)
library(parallel)
library(profvis)
library(here)

source(here('FonctionsSimu.R'))
#source('~/Dropbox/code_BlockCovarianceTest/blocks_loop.R', echo = TRUE)
source(here('Precision_IWT_function.R'), echo=TRUE)
source(here('utilities.R'), echo = TRUE)

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
