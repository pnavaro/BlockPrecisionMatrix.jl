# -*- coding: utf-8 -*-
using Random, LinearAlgebra, Distributions
using BenchmarkTools, GLMNet, Plots, StatsBase

include("../src/PrecisionMatrix.jl")


# +
p        = 20 
n        = 500
b        = 3 
blocsOn  = [[1,3]]
resBlocs = PrecisionMatrix.structure_cov(p, b, blocsOn, seed = 42)
resBlocs[:blocs]

D        = rand(Uniform(1e-4, 1e-2), p)
resmat   = PrecisionMatrix.cov_simu(resBlocs[:blocs], resBlocs[:indblocs], 
                   blocsOn, D)
# -


heatmap(resmat[:CovMat])

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

"""
    stat_test(block1_perm, block2, data_orig, points_x, points_y)

We use the Xia estimator for the precision matrix
"""
function stat_test(block1_perm, block2, 
        data_orig, points_x, points_y)
    
    data.orig[:, points_x] .= block1_perm
    Tprec, TprecStd = prec_xia(data_orig)
    submat = TprecStd[points_x, points_y]
    return sum(submat).^2 
end

""" Permutation test:
  #T_coeff <- array(dim=c(B,p,p)) # permuted test statistics
"""
permute(x :: Array{Float64, 2}, n) = x[randperm(n), :] 

# +
"""
    function SCADmod(yvector, x, lambda)

function returning the fitted values pf regression with 
the SCAD penalty (used for conditional permutations)
"""
function scad_mod(yvector, x, lambda)
  scad = PrecisionMatrix.ncvreg(x, yvector, penalty=:SCAD, lambda=lambda)
    
  n = size(x)[2]
  
  fitted = hcat(ones(n),x) * scad.beta
    
  return fitted

end
# -

"""
    permute_conditional(y, n, data_complement, estimation)

uses SCAD/OLS

"""
function permute_conditional(y, n, data_complement, estimation)
    # SCAD/OLS estimation
    if extimation == :LM
        XX = hcat(ones(n), data_complement)
        beta = pinv(XX) * y
        # do prediction
        fitted = XX * beta
    elseif estimation == :SCAD
        nrows, ncols = size(data_complement)
        λ = 2*sqrt(var(y[:,1]) * log(ncols)/nrows)
        SCAD = [scad_mod(v, data_complement,λ) for v in eachcol(y)]
    end    

    residuals = y .- fitted
    
    return fitted .+ residuals[permutation,:]
end

# +
# tests on rectangles
pval_array = zeros(Float64,(2,p,p))
corrected_pval = zeros(Float64, (p,p))
seeds = round.(100000 * rand(B))

responsible_test = zeros(Float64, (p,p))
ntests_blocks = zeros(Float64,(p,p))
zeromatrix = zeros(Float64,(p,p))

# -

for ix in 2:nblocks  # x coordinate starting point.

    for lx in 0:(nblocks-ix) # length on x axis of the rectangle
        
        index_x = ix:(ix+lx) # index first block
        points_x = which(blocks %in% index.x) # coefficients in block index.x
        data_B1 = data[,points.x] # data of the first block

        data_B1_array = array(data=data.B1,dim=c(n,dim(data.B1)[2],B))
        data_B1_list = lapply(seq(dim(data.B1.array)[3]), function(x) data.B1.array[ , , x])

        for iy in 1:(ix-1) # y coordinate starting point. stops before the diagonal
            for ly in 0:(ix-iy-1) # length on y axis of the rectangle

                # data of the second block
                index_y = iy:(iy+ly) # index second block
                points_y = which(blocks %in% index.y)
                data_B2 = data[,points.y]

                index_complement = (1:nblocks)[-c(index.x,index.y)]
                points_complement = which(blocks %in% index.complement)
                data_complement = data[,points.complement]

                # print(paste0('x:',index.x))
                # print(paste0('y:',index.y))
                # print(paste0('c:', index.complement))

                # permuted data of the first block
                ncol = dim(data.complement)[2]

                if ncol > 0
                  data_B1_perm_l = lapply(data_B1_list,
                                          permute_conditional,
                                          n=n,
                                          data_complement=data.complement,
                                          estimation=estimation)
                else
                  data_B1_perm_l = lapply(data_B1_list,permute,n=n)
                end

                data_B1_perm = simplify2array(data.B1.perm.l)

                testmatrix = zeromatrix
                testmatrix[points.x,points.y] = 1
                ntests_blocks = ntests_blocks + testmatrix
                plot(ntests.blocks,main=paste0('Blocks: ',index.x,'-',index.y))
                plot(testmatrix,main=paste0('Blocks: ',index.x,'-',index.y))

                T0_tmp = stat_test(data.B1,data.B2,data.orig=data,points.x=points.x,points.y=points.y)

                Tperm_tmp = apply(data.B1.perm,3,stat.test,block2=data.B2,
                                  data.orig=data,points.x=points.x,points.y=points.y)
                pval_tmp = mean(Tperm.tmp >= T0.tmp)

                corrected_pval_temp = zeros(Float64,(p,p))
                corrected_pval_temp[points_x,points_y] = pval_tmp
                corrected_pval_temp[points_y,points_x] = pval_tmp # simmetrization

                # old adjusted p-value
                pval_array[1,:,:] .= corrected_pval 
                # p-value resulting from the test of this block
                pval_array[2,:,:] .= corrected_pval_temp 

                # maximization for updating the adjusted p-value
                corrected_pval = maximum(pval_array,dims=1) 

                index = zeros(Int,(p,p))
                for i in 1:p, j in 1:p 
                    index[i,j] = findmax(A[:,i,j])[2] 
                end
                responsible.test[which(index==2)] = paste0(paste0(index.x,collapse=','),'-',paste0(index.y,collapse=','))
            end
        end
    end
end
