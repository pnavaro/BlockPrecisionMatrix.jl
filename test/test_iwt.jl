using RCall


R"""
library(mvtnorm)
library(fields)
library(ncvreg)
library(tictoc)
library(mixAK)
library(glmnet)

p        = 20 
n        = 500 
b        = 3 
blocsOn  = list(c(1,3))

# simulation des donnees
resBlocs = StructureCov(p, b, blocsOn, seed = 2)
D        = runif(p, 10^-4, 10^-2)
resmat   = CovSimu(resBlocs$blocs, resBlocs$indblocs, blocsOn, D = D)

data    = rmvnorm(n, rep(0, p), sigma = resmat$CovMat)
p.part     = sapply(1:length(resBlocs$indblocs), function(i) length(resBlocs$indblocs[[i]]))

(blocks     = rep(1:b, p.part))

B <- 1000
estimation <- 'SCAD'

n <- dim(data)[1]
p = dim(data)[2]
nblocks = length(levels(factor(blocks)))
nblocks

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

# function returning the fitted values pf regression with the SCAD penalty (used for conditional permutations)
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

# tests on rectangles
pval_array = array(dim=c(2,p,p))
corrected.pval = matrix(0,nrow=p,ncol=p)
seeds = round(runif(B,0,1000000))

responsible.test = matrix(nrow=p,ncol=p)
ntests.blocks = zeromatrix = matrix(0,nrow=p,ncol=p)

tic()
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

        index.complement = (1:nblocks)[-c(index.x,index.y)]
        points.complement = which(blocks %in% index.complement)
        data.complement = data[,points.complement]

        # print(paste0('x:',index.x))
        # print(paste0('y:',index.y))
        # print(paste0('c:', index.complement))

        # permuted data of the first block
        ncol = dim(data.complement)[2]

        if(ncol > 0){
          data.B1.perm.l = lapply(data.B1.list,
                                  permute.conditional,
                                  n=n,
                                  data.complement=data.complement,
                                  estimation=estimation)
        }else{
          data.B1.perm.l = lapply(data.B1.list,permute,n=n)
        }

        data.B1.perm = simplify2array(data.B1.perm.l)

        testmatrix = zeromatrix
        testmatrix[points.x,points.y] = 1
        ntests.blocks = ntests.blocks + testmatrix

        T0.tmp = stat.test(data.B1,data.B2,data.orig=data,points.x=points.x,points.y=points.y)

        Tperm.tmp = apply(data.B1.perm,3,stat.test,block2=data.B2,
                          data.orig=data,points.x=points.x,points.y=points.y)
        pval.tmp = mean(Tperm.tmp >= T0.tmp)

        corrected.pval_temp = matrix(0,nrow=p,ncol=p)
        corrected.pval_temp[points.x,points.y] = pval.tmp
        corrected.pval_temp[points.y,points.x] = pval.tmp # simmetrization

        pval_array[1,,] = corrected.pval # old adjusted p-value
        pval_array[2,,] = corrected.pval_temp # p-value resulting from the test of this block

        corrected.pval = apply(pval_array,c(2,3),max) # maximization for updating the adjusted p-value

        index = apply(pval_array,c(2,3),which.max)
        responsible.test[which(index==2)] = paste0(paste0(index.x,collapse=','),'-',paste0(index.y,collapse=','))
      }
    }
  }
}
toc()

responsible.test
corrected.pval
ntests.blocks

"""


