import NCVREG: ncvreg

"""
    function SCADmod(yvector, x, lambda)

function returning the fitted values pf regression with 
the SCAD penalty (used for conditional permutations)

```R
SCADmod = function(yvector,x,lambda){
  regSCAD = ncvreg::ncvreg(x,yvector,
                           penalty='SCAD',lambda=lambda)
  fitted = cbind(1,x) %*% regSCAD\$beta
  return(fitted)
}
```

"""
function scad_mod(yvector, x, λ)
    
  γ = 3.7

  @show x
  beta = ncvreg(x, yvector, λ, :SCAD, γ)
    
  @show size(x)
  @show size(beta)
  n = size(x)[2]
  
  fitted = hcat(ones(n),x) * beta
    
  @show size(fitted)
  return fitted

end

""" 
    permute(x, n) 

Permutation

"""
permute(x :: Array{Float64, 2}, n) = x[randperm(n), :] 



"""
    permute_conditional(y, n, data_complement, estimation)

uses SCAD/OLS

```R
permute.conditional = function(y,n,data.complement,estimation){ 
    # uses SCAD/OLS
  permutation = sample(n)

  # SCAD/OLS estimaytion
  fitted = switch(estimation,
                  LM=lm.fit(cbind(1,data.complement),y)\$fitted,
                  SCAD=apply(y,2,SCADmod,x=data.complement,
                             lambda=2*sqrt(var(y[,1])*log(ncol(data.complement))/nrow(data.complement))))

  residuals = y - fitted

  result = fitted + residuals[permutation,]
  return(result)
}
```

"""
function permute_conditional(y, n, data_complement, estimation)

    # SCAD/OLS estimation
    if estimation == :LM
        XX = hcat(ones(n), data_complement)
        beta = pinv(XX) * y
        # do prediction
        fitted = XX * beta
    elseif estimation == :SCAD
        nrows, ncols = size(data_complement)
        λ = 2*sqrt(var(y[:,1]) * log(ncols)/nrows)
        fitted = [scad_mod(v, data_complement,λ) for v in eachcol(y)]
    end    

    @show size(y)
    @show size(fitted)
    residuals = y .- fitted
    
    return fitted .+ residuals[permutation,:]

end
