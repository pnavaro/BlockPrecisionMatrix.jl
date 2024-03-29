# Function estimating the precision matrix with the method from Xia et al (code Madison)

```R
library(glmnet)

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
```


# Function performing the test on blocks and adjusting the results

- inputs:
    - data: data matrix (n*p) where n is the sample size and p the number of grid points (for now. it will be spline coefficients?)
    - blocks: vector of length p containing block indexes (numeric, from 1 to number of blocks) for each grid point
    - B: number of permutations done to evaluate the tests p-values

```R
IWT_Block_precision <- function(data,blocks,B=1000,estimation='SCAD'){  
  
  n <- dim(data)[1]
  p = dim(data)[2]
  nblocks = length(levels(factor(blocks)))
  
  stat.test = function(block1.perm,block2,data.orig,points.x,points.y){
    data.orig[,points.x] = block1.perm
    PrecMat = PrecXia(data.orig)
    #Rhohat = cor(block1,block2,method='pearson')
    #Rohat.std = Rhohat^2/(1-Rhohat^2)
    submat = PrecMat$TprecStd[points.x,points.y]
    return(sum(submat)^2) # We use the Xia estimator for the precision matrix
  }
  
  # Permutation test:
  #T_coeff <- array(dim=c(B,p,p)) # permuted test statistics
  permute = function(x,n){
    permutation = sample(n)
    result = x[permutation,] 
    return(result) 
  }
  permute.conditional = function(y,n,data.complement,estimation){ # uses SCAD/OLS
    permutation = sample(n)
    
    # SCAD/OLS estimaytion
    fitted = switch(estimation, 
                    LM=lm.fit(cbind(1,data.complement),y)$fitted,
                    SCAD=apply(y,2,SCADmod,x=data.complement,lambda=2*sqrt(var(y[,1])*log(ncol(data.complement))/nrow(data.complement))))
    
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
  
  # For parallelization: the loops are independent between each other, so they can be easily parallelized. 
  # The only point to pay attention to is that for computing the adjusted p-value, the maximization is done at the end of each loop.
  # It would also be possible to save all the results of each loop instead (in an array) and do the maximization only once at the end.
  for(ix in 2:nblocks){ # x coordinate starting point. 
    for(lx in 0:(nblocks-ix)){ # length on x axis of the rectangle
      index.x = ix:(ix+lx) # index first block
      points.x = which(blocks %in% index.x) # coefficients in block index.x
      data.B1 = data[,points.x] # data of the first block
      
      data.B1.array = array(data=data.B1,dim=c(n,dim(data.B1)[2],B)) # tableau de copies de data.B1
      data.B1.list = lapply(seq(dim(data.B1.array)[3]), function(x) data.B1.array[ , , x]) # liste de copies de data.B1
      
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
            # A chaque élément de la liste de copies de data.B1.list, 
            # on applique permute.conditional
            data.B1.perm.l = lapply(data.B1.list,permute.conditional,n=n,data.complement=data.complement,estimation=estimation)
          }else{
            data.B1.perm.l = lapply(data.B1.list,permute,n=n)
          }
          
          data.B1.perm = simplify2array(data.B1.perm.l)

          testmatrix = zeromatrix
          testmatrix[points.x,points.y] = 1
          ntests.blocks = ntests.blocks + testmatrix
          image.plot(ntests.blocks,main=paste0('Blocks: ',index.x,'-',index.y))
          image.plot(testmatrix,main=paste0('Blocks: ',index.x,'-',index.y))
          # Estimation de la statistique de test pour les 2 blocs considérés
          T0.tmp = stat.test(data.B1,data.B2,data.orig=data,points.x=points.x,points.y=points.y)
          # Estimation de la statistique de test pour chacune des parmutations
          Tperm.tmp = apply(data.B1.perm,3,stat.test,block2=data.B2,data.orig=data,points.x=points.x,points.y=points.y)
          # calcul de la p-value
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
  return(list(corrected.pval=corrected.pval,responsible.test=responsible.test,ntests.blocks=ntests.blocks))
}

```
