# -*- coding: utf-8 -*-
using Random, LinearAlgebra, Distributions, 
using BenchmarkTools, GLMNet, Plots

# +
include("src/utils.jl")
include("src/rotation_matrix.jl")
include("src/fonctions_simu.jl")
p        = 20 
n        = 500
b        = 3 
blocsOn  = [[1,3]]
resBlocs = structure_cov(p, b, blocsOn, seed = 42)
resBlocs[:blocs]

D        = rand(Uniform(1e-4, 1e-2), p)
resmat   = cov_simu(resBlocs[:blocs], resBlocs[:indblocs], 
                   blocsOn, D)
# -


heatmap(resmat[:CovMat])

plotlyjs()

heatmap(resmat[:CovMat], c=ColorGradient([:red,:blue]))

heatmap(resmat[:PreMat])

data = rand!( MvNormal(resmat[:CovMat]), zeros(Float64,(p, n)));


p_part = map( length,  resBlocs[:indblocs])


blocks  =vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...);

B = 1000
estimation = :SCAD

n, p  = size(data)


# +
using CategoricalArrays

nblocks = length(levels(CategoricalArray(blocks)))

# -

function stat_test(block1_perm, block2, 
        data_orig, points_x, points_y)
    
    data.orig[:, points_x] .= block1_perm
    Tprec, TprecStd = prec_xia(data_orig)
    #Rhohat = cor(block1,block2,method='pearson')
    #Rohat.std = Rhohat^2/(1-Rhohat^2)
    submat = TprecStd[points_x, points_y]
    return sum(submat).^2 # We use the Xia estimator for the precision matrix
end

""" Permutation test:
  #T_coeff <- array(dim=c(B,p,p)) # permuted test statistics
"""
permute(x :: Array{Float64, 2}, n) = x[randperm(n), :] 

# function returning the fitted values pf regression with the SCAD penalty (used for conditional permutations)
function SCADmod(yvector, x, lambda)
  regSCAD = ncvreg::ncvreg(x,yvector,penalty='SCAD',lambda=lambda)
  fitted = cbind(1,x) %*% regSCAD$beta
  return(fitted)
end

"""
# uses SCAD/OLS
    permutation = randperm(n)
"""
function permute_conditional(y, n, data_complement, estimation)
    # SCAD/OLS estimaytion
    if extimation == :LM
        fitted = lm.fit(cbind(1,data_complement),y)$fitted
    elseif estimation == :SCAD
        nrows, ncols =. size(data_complement)
        λ=2*sqrt(var(y[:,1])*log(ncols)/nrows)))
        SCAD=apply(y,2,SCADmod, x=data_complement, lambda=λ)
    end    

    residuals = y .- fitted
    
    return fitted .+ residuals[permutation,:]
end


