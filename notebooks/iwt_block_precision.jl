# -*- coding: utf-8 -*-
using Random, LinearAlgebra, Distributions
using BenchmarkTools, GLMNet, Plots, StatsBase
using InvertedIndices

include("../src/ncvreg.jl")
include("../src/rotation_matrix.jl")
include("../src/fonctions_simu.jl")
include("../src/prec_xia.jl")
include("../src/precision_iwt.jl")

# +
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

heatmap(resmat[:CovMat], c=ColorGradient([:red,:blue]))

heatmap(resmat[:PreMat])

d = MvNormal(resmat[:CovMat])
rng = MersenneTwister(1234)
data = rand!(rng, d , zeros(Float64,(p, n)));


p_part = map( length,  resBlocs[:indblocs])


blocks  =vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...)

B = 1000
estimation = :SCAD

n, p  = size(data)


# +
using CategoricalArrays

nblocks = length(levels(CategoricalArray(blocks)))


# +
using Random, GLMNet, InvertedIndices
using Statistics, Distributions
using LinearAlgebra

function prec_xia( X :: Array{Float64, 2} )
    
    nrows, ncols = size(X)

    betahat = zeros(Float64,(ncols-1,ncols))
    reshat  = copy(X)
    
    for k in 1:ncols
      y = view(X, :, k )
      λ = [2*sqrt(var(y)*log(ncols)/nrows)]
      fitreg = glmnet(X[:,Not(k)], y, lambda = λ, standardize=false)
      betahat[:,k] .= vec(fitreg.betas)
      reshat[:,k]  .= X[:,k] .- vec(predict(fitreg,X[:,Not(k)]))
    end
    
    rtilde = cov(reshat) .* (nrows-1) ./ nrows
    rhat   = rtilde
    
    for i in 1:ncols-1
       for j in (i+1):ncols
        rhat[i,j] = -(rtilde[i,j]+rtilde[i,i]*betahat[i,j]+rtilde[j,j]*betahat[j-1,i])
        rhat[j,i] = rhat[i,j]
      end
    end
    
    Tprec    = copy(1 ./ rhat)
    TprecStd = copy(Tprec)
    
    for i in 1:(ncols-1)
        for j in (i+1):ncols
            Tprec[i,j] = Tprec[j,i] = rhat[i,j]/(rhat[i,i]*rhat[j,j])
            thetahatij = (1+(betahat[i,j]^2  * rhat[i,i]/rhat[j,j]))/(nrows*rhat[i,i]*rhat[j,j])   
            TprecStd[i,j] = TprecStd[j,i] = Tprec[i,j]/sqrt(thetahatij)
        end
    end

    Tprec, TprecStd
end


# +
"""
    stat_test(block1_perm, block2, data_orig, points_x, points_y)

We use the Xia estimator for the precision matrix
"""
function stat_test(block1_perm, block2, data_orig, points_x, points_y)
    
    data_orig[:, points_x] .= block1_perm
    Tprec, TprecStd = prec_xia(data_orig)
    submat = TprecStd[points_x, points_y]
    return sum(submat).^2
    
end
# -

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
    
  scad = ncvreg(x, yvector, penalty=:SCAD, lambda=lambda)
    
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
B
# -

index = zeros(Int,(p,p))
corrected_pval_temp = zeros(Float64,(p,p));

ix = 2
lx = 0
index_x = ix:(ix+lx)

points_x = findall(x -> x in index_x, blocks)

data_B1 = data[:,points_x]

data_B1_array = reshape(data_B1,n,size(data_B1)[2],B)

# +

for ix in 2:nblocks  # x coordinate starting point.

    for lx in 0:(nblocks-ix) # length on x axis of the rectangle
        
        index_x = ix:(ix+lx) # index first block
        points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
        data_B1 = data[:,points_x] # data of the first block
        data_B1_array = reshape(data_B1,n,size(data_B1)[2],B)
        data_B1_list = [data_B1_array[:,:,i] for i in 1:size(data_B1_array)[3]]

        for iy in 1:(ix-1) # y coordinate starting point. stops before the diagonal
            for ly in 0:(ix-iy-1) # length on y axis of the rectangle

                # data of the second block
                index_y = iy:(iy+ly) # index second block
                points_y = findall( x -> x in index_y, blocks)
                data_B2 = data[:,points_y]

                index_complement = collect(1:nblocks)[Not([index_x;index_y])]
                points_complement = findall( x -> x in index_complement, blocks)
                data_complement = data[:,points_complement]

                println("x:$index_x")
                println("y:$index_y")
                println("c:$index_complement")

                # permuted data of the first block
                ncol = size(data_complement)[2]

                if ncol > 0
                  data_B1_perm_l = [permute_conditional(B1, n, data_complement, estimation) for B1 in data_B1_list]
                else
                  data_B1_perm_l = [permute(B1,n) for B1 in data_B1_list]
                end

                data_B1_perm = 
                
                for (k,m) in enumerate(data_B1_perm_l) 
                    data_B1_perm[:,:,k] = m 
                end

                testmatrix = zeromatrix
                testmatrix[points_x,points_y] = 1
                ntests_blocks = ntests_blocks + testmatrix
                
                heatmap(ntests_blocks, title="Blocks: $index_x - $index_y")
                heatmap(testmatrix,    title="Blocks: $index_x - $index_y")

                T0_tmp = stat_test(data_B1,data_B2,data,points_x,points_y)

                Tperm_tmp = []
                for k in 1:size(data_B1_perm)[3]
                    push!(Tperm_tmp,stat_test(data_B1_perm[:,:,k],data_B2, data, points_x, points_y))
                end
                
                
                pval_tmp = mean(Tperm_tmp .>= T0_tmp)

                
                corrected_pval_temp[points_x,points_y] = pval_tmp
                corrected_pval_temp[points_y,points_x] = pval_tmp # symmetrization

                # old adjusted p-value
                pval_array[1,:,:] .= corrected_pval 
                # p-value resulting from the test of this block
                pval_array[2,:,:] .= corrected_pval_temp 

                # maximization for updating the adjusted p-value
                corrected_pval = maximum(pval_array,dims=1) 

                for i in 1:p, j in 1:p 
                    index[i,j] = findmax(A[:,i,j])[2] 
                end
                responsible_test[which(index==2)] = "$(index_x) - $(index_y)"
            end
        end
    end
end
# -


