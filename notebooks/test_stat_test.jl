using Random
using RCall
using Test

ref = R"""
rm(list=ls())

library(dotCall64)
library(grid)
library(spam)
library(Matrix)
library(colorspace)
library(lme4)
library(maps)
library(glmnet)
library(mvtnorm)
library(fields)
library(parallel)
library(profvis)
library(ncvreg)
library(tictoc)
library(mixAK)

# fonction pour definir les indices des blocs
StructureCov = function(p, b = 10, blocsOn, seed = p) {
  # p : dimension des donnees
  # b : nombre de blocs
  # blocsOn : liste des indices des couples de blocs "allumes"
  
  set.seed(seed)
  
  blocs = sample(3:(p-2), size = (b-1))
  blocs = sort(blocs)
  blocs = c(1, blocs, p)
  indblocs = list()
  indblocs[[1]] = 1:blocs[2]
  for(i in 2:(length(blocs)-1)) {
    indblocs[[i]] = (blocs[i]+1):blocs[i+1]
  }
  
  matBlocs = matrix(0, ncol = p, nrow = p)
  for (l in 1:length(indblocs)){
    matBlocs[indblocs[[l]], indblocs[[l]]] = 1
  }
  
  for(k in 1:length(blocsOn)) {
    for (i in indblocs[[blocsOn[[k]][1]]]) {
      for (j in indblocs[[blocsOn[[k]][2]]]) {
        matBlocs[i, j] = 1
        matBlocs[j, i] = 1
      }
    }
  }
  
  return(list(blocs = blocs, indblocs = indblocs, matBlocs = matBlocs, blocsOn = blocsOn))
}


# fonction pour simuler des matrices de covariances avec blocs
CovSimu = function(blocs, indblocs, blocsOn, D) {
  # blocs    : indice des intervalles separant les blocs (sortie de StructureCov)
  # indblocs : liste des indices des blocs (sortie de StructureCov)
  # blocsOn  : liste des indices des couples de blocs "allumes"
  # D        : vecteur des valeurs propres de la matrice a simuler
  
  set.seed(p)
  
  b = length(blocs) - 1
  p = length(unlist(indblocs))
  P = matrix(0, ncol = p, nrow = p)
  
  # matrices de rotation pour les blocs "allumes"
  for (i in 1:length(blocsOn)){
    B1 = indblocs[[blocsOn[[i]][1]]][1]:indblocs[[blocsOn[[i]][1]]][length(indblocs[[blocsOn[[i]][1]]])]
    B2 = indblocs[[blocsOn[[i]][2]]][1]:indblocs[[blocsOn[[i]][2]]][length(indblocs[[blocsOn[[i]][2]]])]
    nB = length(indblocs[[blocsOn[[i]][1]]])+length(indblocs[[blocsOn[[i]][2]]])
    P[c(B1, B2), c(B1, B2)] = rRotationMatrix(1, nB)
  }
  
  # matrices de rotation pour les autre blocs centraux (non allumes)
  for (i in setdiff(1:b, unlist(blocsOn))) {
    P[indblocs[[i]][1]:indblocs[[i]][length(indblocs[[i]])], indblocs[[i]][1]:indblocs[[i]][length(indblocs[[i]])]] = rRotationMatrix(1, length(indblocs[[i]]))
  }
  
  for (i in 1:length(blocsOn)){
    P[indblocs[[blocsOn[[i]][1]]][1]:indblocs[[blocsOn[[i]][1]]][length(indblocs[[blocsOn[[i]][1]]])] , indblocs[[blocsOn[[i]][2]]][1]:indblocs[[blocsOn[[i]][2]]][length(indblocs[[blocsOn[[i]][2]]])]] = 0
  }
  P = t(P)
  
  # Matrice de covariance
  CovMat = t(P)%*%diag(D)%*%P
  
  # Matrice de precision
  PreMat = t(P)%*%diag(1/D)%*%P
  
  return(list(CovMat = CovMat, PreMat = PreMat))
}

p        = 20 # c(20, 100)
n        = 500 # c(100, 200, 500, 1000)
b        = 3 # c(3, 5, 5, 20)
blocsOn  = list(c(1,3))

# simulation des donnees
resBlocs = StructureCov(p, b, blocsOn, seed = 2)
D        = runif(p, 10^-4, 10^-2)
resmat   = CovSimu(resBlocs$blocs, resBlocs$indblocs, blocsOn, D = D)

data    = rmvnorm(n, rep(0, p), sigma = resmat$CovMat)
p.part  = sapply(1:length(resBlocs$indblocs), function(i) length(resBlocs$indblocs[[i]]))
blocks  = rep(1:b, p.part)

# +

PrecXia = function(X){
  n = nrow(X)
  betahat = matrix(0,ncol(X)-1,ncol(X))
  reshat  = X
  for (k in 1:ncol(X)){
    fitreg  = glmnet(X[,-k],X[,k],family="gaussian",lambda = 2*sqrt(var(X[,k])*log(ncol(X))/nrow(X)),standardize = FALSE)
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

B <- 1000
estimation <- 'SCAD'

nblocks = length(levels(factor(blocks)))

stat.test = function(block1.perm,block2,data.orig,points.x,points.y){
  data.orig[,points.x] = block1.perm
  PrecMat = PrecXia(data.orig)
  #Rhohat = cor(block1,block2,method='pearson')
  #Rohat.std = Rhohat^2/(1-Rhohat^2)
  submat = PrecMat$TprecStd[points.x,points.y]
  return(sum(submat)^2) # We use the Xia estimator for the precision matrix
}

permute = function(x,n){
  permutation = sample(n)
  result = x[permutation,]
  return(result)
}

SCADmod = function(yvector,x,lambda){
  regSCAD = ncvreg::ncvreg(x,yvector,
                           penalty='SCAD',lambda=lambda)
  fitted = cbind(1,x) %*% regSCAD$beta
  return(fitted)
}

permute.conditional = function(y,n,data.complement,estimation){ 
  
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

ref <- vector()

for(ix in 2:nblocks){ # x coordinate starting point.
  for(lx in 0:(nblocks-ix)){ # length on x axis of the rectangle
      
    index.x = ix:(ix+lx) # index first block
    points.x = which(blocks %in% index.x) # coefficients in block index.x
    data.B1 = data[,points.x] # data of the first block

    data.B1.array = array(data=data.B1,dim=c(n,dim(data.B1)[2],B))
    data.B1.list = lapply(seq(dim(data.B1.array)[3]), function(x) data.B1.array[ , , x])

    for(iy in 1:(ix-1)){ # y coordinate starting point. stops before the diagonal
      for(ly in 0:(ix-iy-1)){ # length on y axis of the rectangle
      
        # data of the second block
        index.y = iy:(iy+ly) # index second block
        points.y = which(blocks %in% index.y)
        data.B2 = data[,points.y]

        ref <- c(ref, stat.test(data.B1,data.B2,data.orig=data,points.x=points.x,points.y=points.y))

        }
      }
   }
}

ref

"""

import BlockPrecisionMatrix:StatTest

n = 500
p = 20

rng = MersenneTwister(12)

blocs_on  = [[1,3]]

blocks = rcopy(R"blocks")
nblocks = rcopy(R"nblocks")
data = rcopy(R"data")

T0_tmp = Float64[]

stat_test = StatTest(n, p)

# x coordinate starting point and length on x axis of the rectangle
for ix in 2:nblocks, lx in 0:(nblocks-ix)

    # FIRST BLOCK

    index_x = ix:(ix+lx) # index first block
    points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
                  

    # y coordinate starting point. stops before the diagonal and length on y axis of the rectangle
    for iy in 1:(ix-1), ly in 0:(ix-iy-1)

        # SECOND BLOCK

        index_y = iy:(iy+ly) # index second block
        points_y = findall( x -> x in index_y, blocks)

        data_B1 = view(data,:,points_x)
        data_B2 = view(data,:,points_y)

        push!(T0_tmp, stat_test(data_B1, data, points_x, points_y))

    end
end

@test T0_tmp ≈ rcopy(ref)
