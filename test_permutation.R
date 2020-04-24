library(ncvreg)
library(fields)
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

set.seed(1111)

n = 100
Z = matrix(rnorm(5*n),n,5) # data complement
X = matrix(rnorm(3*n),n,3)
Y = cbind(X%*%matrix(c(1,1,1),3,1)+rnorm(n)*.1,Z+rnorm(5*n)*.1)

M = cbind(Y,X,Z)

res = permute.conditional(Y,n,Z,"SCAD")

M1 = cbind(res,X,Z)

par(mfrow = c(1,2),mar = c(3,3,3,3))
image.plot(solve(cov(M)))
image.plot(solve(cov(M1)))

sum(eigen(solve(cov(M1)))$values) 

# [1] 31.91363

sum(eigen(solve(cov(M)))$values) 

# [1] 2346.617

sum(apply(M,2,var)) # somme des variances  des variables qui composent M
# [1] 24.05086

abs( sum(eigen(solve(cov(M1)))$values)-sum(apply(M,2,var)))
