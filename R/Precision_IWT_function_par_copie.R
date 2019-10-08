
library(glmnet)
library(doParallel)
library(foreach)
library(parallel)
library(ncvreg)



# function performing the test on blocks and adjusting the results
# inputs:
# data: data matrix (n*p) where n is the sample size and p the number of grid points (for now. it will be spline coefficients?)
# blocks: vector of length p containing block indexes (numeric, from 1 to number of blocks) for each grid point
# B: number of permutations done to evaluate the tests p-values
source("~/Dropbox/projet spectres/FDA-VisiteA.Pini/code_BlockCovarianceTest/blocks_loop.R")
source("~/Dropbox/projet spectres/FDA-VisiteA.Pini/code_BlockCovarianceTest/utilities.R")
# Test --------------------------------------------------------------------
p.part = c(10,10,10,10,10,10) # size of the blocks: constant for now
p = sum(p.part)
nblocks = length(p.part)

n = 200 # sample size
blocks = rep(1:nblocks,p.part)

s = rep(1,nblocks) # variance in each block
Sigma = matrix(0,nrow=p,ncol=p) # covariance matrix

cov.block = 3 # covariance in the block where the covariance is non zero

for(i in 1:p){
  for(j in 1:p){
    if(blocks[i]==blocks[j]){
      Sigma[i,j] = 5*s[blocks[i]]*exp(-(i-j)^2/(p))
    }
    if(blocks[i]==2 & blocks[j]==3 | blocks[i]==3 & blocks[j]==2 ){
      Sigma[i,j] = cov.block
    }
    # with another block
    if(blocks[i]==3 & blocks[j]==6 | blocks[i]==6 & blocks[j]==3 ){
      Sigma[i,j] = cov.block
    }
  }
}
Sigma = Sigma + diag(rep(5,p))

mu = rep(0,p)
mu[which(blocks==1|blocks==3)] = 2

library(fields)
image.plot(Sigma)

EigSigma = eigen(Sigma)
InvSigma = EigSigma$vectors %*% diag(1/EigSigma$values) %*% t(EigSigma$vectors)

image.plot(InvSigma)

library(mvtnorm)
data = rmvnorm(n,mu,Sigma)
matplot(t(data),type='l') # instance of generated data

S = var(data)
invS = solve(S)
image.plot(invS)

Xiaest = PrecXia(data)
image.plot(Xiaest$TprecStd)


# sequential --------------------------------------------------------------

tic = proc.time()
resultseq = IWT_Block_precision(data,blocks,B=100,nworkers=1)
tsequential = proc.time()-tic

# parallel --------------------------------------------------------------

tic = proc.time()
resultpar = IWT_Block_precision(data,blocks,B=100,nworkers=detectCores()-1,estimation='LM')
tparallel1 = proc.time()-tic

image.plot(resultpar)
image.plot(resultseq)

tsequential/60
tparallel1/60




# LM vs. SCAD -------------------------------------------------------------
data = rmvnorm(20,mu,Sigma)
resultLM = IWT_Block_precision(data,blocks,B=100,nworkers=detectCores()-1,estimation='LM')

resultSCAD = IWT_Block_precision(data,blocks,B=100,nworkers=detectCores()-1,estimation='SCAD')
# LM does not find differences because it cannot estimate the residuals

image.plot(resultLM)
image.plot(resultSCAD)

data = rmvnorm(60,mu,Sigma)
resultLM = IWT_Block_precision(data,blocks,B=100,nworkers=detectCores()-1,estimation='LM')

resultSCAD = IWT_Block_precision(data,blocks,B=100,nworkers=detectCores()-1,estimation='SCAD')


image.plot(resultLM)
image.plot(resultSCAD)


data = rmvnorm(200,mu,Sigma)
resultLM = IWT_Block_precision(data,blocks,B=100,nworkers=detectCores()-1,estimation='LM')

resultSCAD = IWT_Block_precision(data,blocks,B=100,nworkers=detectCores()-1,estimation='SCAD')
resultSCAD_unadjusted = IWT_Block_precision_unadjusted(data,blocks,B=100,nworkers=detectCores()-1,estimation='SCAD')

image.plot(resultLM)

layout(1:2)
image.plot(resultSCAD)
image.plot(resultSCAD_unadjusted)


image.plot(resultLM<0.05)
image.plot(resultSCAD<0.05)

