using RCall

# somme les valeurs propres de l'inverse de la covariance de M1

S1 = R"""
library(ncvreg)

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

set.seed(3)

n = 100
Z = matrix(rnorm(500),n,5) # data complement
X = matrix(rnorm(300),n,3)
Y = cbind(X%*%matrix(c(1,1,1),3,1),Z+rnorm(500)*.1)

M = cbind(Y,X,Z)
M = M + diag(M)

res = permute.conditional(Y,n,Z,"SCAD")

M1 = cbind(res,X,Z)

sum(eigen(solve(cov(M1)))$values) 
"""

# [1] 31.91363

@show S1

# somme les valeurs propres de l'inverse de la covariance de M
S2 = R"""
sum(eigen(solve(cov(M)))$values) 
"""

# [1] 2346.617

@show S2

S3 = R"""
sum(apply(M,2,var)) # somme des variances  des variables qui composent M
"""
# [1] 24.05086
@show S3

# après la permutation la somme les valeurs propres de l'inverse de la 
# covariance de M1 est du même ordre que somme des variances  des variables qui composent M
# on purrait donc tester si

test_covariance = R"abs( sum(eigen(solve(cov(M1)))$values)-sum(apply(M,2,var)))<10"

@testset "Permutation" begin

@test rcopy(test_covariance)

end



