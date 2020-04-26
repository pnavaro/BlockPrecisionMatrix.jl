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

  beta = NCVREG.ncvreg(x, yvector, λ, :SCAD, γ)
    
  n = size(x)[1]
  xx = hcat(ones(n),x) 
  
  fitted = xx * beta[:,1:length(λ)]
    
  return fitted

end

""" 
    permute(x, n) 

Permutation

"""
function permute(rng, x :: Array{Float64, 2}, n) 
   x[randperm(rng, n), :] 
end

export permute_conditional

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

    @show size(y)

    fitted = similar(y)
  
    if estimation == :LM
        XX = hcat(ones(n), data_complement)
        beta = pinv(XX) * y
        fitted .= XX * beta
    elseif estimation == :SCAD
        nrows, ncols = size(data_complement)
        λ = 2*sqrt(var(y[:,1]) * log(ncols)/nrows)
        fitted .= hcat([scad_mod(v, data_complement,λ) for v in eachcol(y)]...)
    end    

    @show size(fitted)

    residuals = y .- fitted
    n = size(y)[1]
    permutation = randperm(n)
    
    return fitted .+ residuals[permutation,:]

end

function permute_conditional(rng :: AbstractRNG, Y, Z)
    row_y, col_y = size(Y)
    row_z, col_z = size(Z)
    
    @assert row_y == row_z
    
    fitted = similar(Y)
    for j in 1:col_y
        y = Y[:,j]
        λ = 2*sqrt(var(Y[:,1])*log(col_z)/row_z)
        beta = NCVREG.coef(NCVREG.SCAD(Z, y, [λ]))  
        fitted[:,j] .= vec(hcat(ones(row_z),Z) * beta)
    end
    
    residuals = Y .- fitted
    
    permutation = randperm(rng, row_y)
        
    return fitted .+ residuals[permutation,:]
end
