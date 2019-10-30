"""
    scad_mod(yvector,x,lambda)

function returning the fitted values pf regression with the SCAD penalty (used for conditional permutations)
"""
function scad_mod(yvector, x, lambda)
  regSCAD = ncvreg(x, yvector, penalty= :SCAD, lambda=lambda)
  n, p = size(re
  fitted  = hcat(ones(regSCAD$beta
  fitted
end

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
  #submat = PrecMat$TprecStd[points.x,points.y]
  submat = PrecMat[points.x,points.y]
  return(sum(submat)^2) # We use the Xia estimator for the precision matrix
}

