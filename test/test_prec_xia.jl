using LinearAlgebra
using PrecisionMatrix
using RCall
using Test
R"library(glmnet)"

nrows, ncols = 8, 5


@rput ncols
@rput nrows

X_R = R"""
    X0 <- matrix(1:nrows*ncols, nrows, ncols)
    Sigmainv <- .25^abs(outer(1:ncols,1:ncols,"-"))
    backsolve(chol(Sigmainv), X0)
"""

X0 = repeat(1:nrow, 1, ncols) .* ncols

σ_inv = .25 .^ (abs.( collect(1:ncols) .- collect(1:ncols)'))

X_jl = cholesky(σ_inv).U \ X0[1:ncols,:]

@test X_jl ≈ rcopy(X_R)


beta_R = R"""
k <- 1
fitreg  = glmnet::glmnet(X[,-k],X[,k], family="gaussian",
                             standardize = FALSE)
fitreg$beta
"""

#=
```{r}
lambda = rep(0,ncol(X))
for (k in 1:ncol(X)) {
    a = 2 * (ncol(X)-k+1) / ncol(X)
    lambda[k] = a*sqrt(var(X[,k])*log(ncol(X))/nrow(X))
    }

lambda
```

```{r}
for (k in 1:ncol(X)) {
fitreg  = glmnet::glmnet(X[,-k],X[,k], family="gaussian",
                         lambda = lambda,
                             standardize = FALSE)
print(fitreg$beta[1])
}
```

```{r}
PrecXia = function(X){
  n = nrow(X)
  betahat = matrix(0,ncol(X)-1,ncol(X))
  reshat  = X
  for (k in 1:ncol(X)){
    fitreg  = glmnet::glmnet(X[,-k],X[,k],family="gaussian",
        lambda = 2*sqrt(var(X[,k])*log(ncol(X))/nrow(X)),
                             standardize = FALSE) 
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

```{r}
result <- PrecXia(X)
```

```{r}
result
```

```{r}
var(X[,1])
```

```{r}

```
=#
