# function returning the fitted values pf regression with the SCAD penalty (used for conditional permutations)
SCADmod = function(yvector,x,lambda){
  regSCAD = ncvreg::ncvreg(x,yvector,penalty='SCAD',lambda=lambda)
  fitted = cbind(1,x) %*% regSCAD$beta
  return(fitted)
}

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

stat.test = function(block1.perm,block2,data.orig,points.x,points.y){
  data.orig[,points.x] = block1.perm
  PrecMat = PrecXia(data.orig)
  #Rhohat = cor(block1,block2,method='pearson')
  #Rohat.std = Rhohat^2/(1-Rhohat^2)
  submat = PrecMat$TprecStd[points.x,points.y]
  return(sum(submat)^2) # We use the Xia estimator for the precision matrix
}

PrecXia = function(X){
  n = nrow(X)
  betahat = matrix(0,ncol(X)-1,ncol(X))
  reshat  = X
  for (k in 1:ncol(X)){
    fitreg  = glmnet::glmnet(X[,-k],X[,k],family="gaussian",lambda = 2*sqrt(var(X[,k])*log(ncol(X))/nrow(X)),standardize = FALSE) 
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
