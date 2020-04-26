# -*- coding: utf-8 -*-
using Pkg
pkg"add https://github.com/pnavaro/NCVREG.jl"

using CategoricalArrays
using Random
using LinearAlgebra
using Distributions
using GLMNet
using InvertedIndices
using NCVREG
using ProgressMeter
using InvertedIndices
using Statistics

include("../src/rotation_matrix.jl")
include("../src/fonctions_simu.jl")

# +
p = 20 
n = 500
b = 3
rng = MersenneTwister(42)

function generate_data(rng, p, n, b)
    blocs_on  = [[1,3]]
    blocs, indblocs = structure_cov(rng, p, b, blocs_on)
    D = rand(rng, Uniform(1e-4, 1e-2), p)
    covmat, premat   = cov_simu(blocs, indblocs, blocs_on, D)
    #hm = plot(layout=(1,2))
    #heatmap!(hm[1,1], covmat, c=ColorGradient([:red,:blue]), aspect_ratio=:equal)
    #heatmap!(hm[1,2], premat, aspect_ratio=:equal)
    d = MvNormal(covmat)
    p_part = map( length,  indblocs)
    blocks  = vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...)
    data = rand!(rng, d, zeros(Float64,(p, n)))
    data, blocks
end

# -

data, blocks = generate_data(rng, p, n, b)

# +
function permute!(rng, x :: Array{Float64, 2}) 
   n = size(x)[1]
   x .= x[randperm(rng, n), :] 
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


# +
function stat_test(block1_perm, block2, data_orig, points_x, points_y)
    
    data_orig[:, points_x] .= block1_perm
    nrows, ncols = size(data_orig)

    betahat = zeros(Float64,(ncols-1,ncols))
    reshat  = copy(data_orig)
    
    for k in 1:ncols
        x = view(data_orig, :, Not(k))
        y = view(data_orig, :, k )
        λ = [2*sqrt(var(y)*log(ncols)/nrows)]
        fitreg = glmnet(x, y, lambda = λ, standardize=false)
        betahat[:,k] .= vec(fitreg.betas)
        reshat[:,k]  .= y .- vec(GLMNet.predict(fitreg,x))
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
    
    for i in 1:(ncols-1), j in (i+1):ncols
        Tprec[i,j] = rhat[i,j]/(rhat[i,i]*rhat[j,j])
        Tprec[j,i] = Tprec[i,j]
        thetahatij = (1+(betahat[i,j]^2  * rhat[i,i]/rhat[j,j]))/(nrows*rhat[i,i]*rhat[j,j])   
        TprecStd[i,j] = Tprec[i,j]/sqrt(thetahatij)
        TprecStd[j,i] = TprecStd[i,j] 
    end

    submat = TprecStd[points_x, points_y]
    return sum(submat).^2
    
end
# -

function iwt_block_precision(data, blocks; B=1000)
    
    
    estimation = :SCAD
    p, n  = size(data)
    nblocks = length(levels(CategoricalArray(blocks)))
    # tests on rectangles
    pval_array = zeros(Float64,(2,p,p))
    corrected_pval = zeros(Float64, (p,p))
    responsible_test = Array{String,2}(undef,p,p)
    ntests_blocks = zeros(Float64,(p,p))
    zeromatrix = zeros(Float64,(p,p))
    testmatrix = zeros(Float64,(p,p))
    index = zeros(Int,(p,p))
    corrected_pval_temp = zeros(Float64,(p,p))
    Tperm_tmp = zeros(Float64, B)

    # x coordinate starting point and length on x axis of the rectangle
    # for ix in 2:nblocks, lx in 0:(nblocks-ix)
    for ix in 2:2, lx in 0:0
        # FIRST BLOCK

        index_x = ix:(ix+lx) # index first block
        points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
        data_B1 = data[:,points_x] # data of the first block
        data_B1_array = Iterators.repeated(data_B1, B)

        # y coordinate starting point. stops before the diagonal and length on y axis of the rectangle
        # for iy in 1:(ix-1), ly in 0:(ix-iy-1)
        for iy in 1:1, ly in 0:0
            # SECOND BLOCK

            index_y = iy:(iy+ly) # index second block
            points_y = findall( x -> x in index_y, blocks)
            data_B2 = view(data,:,points_y)
            
            @show nblocks
            @show index_x
            @show index_y
            index_complement = collect(filter(index -> !(index in vcat(index_x,index_y)), 1:nblocks))
            points_complement = findall( x -> x in index_complement, blocks)
            data_complement = data[:,points_complement]

            # permuted data of the first block
            ncol = size(data_complement)[2]

            if ncol > 0
                data_B1_perm = [permute_conditional(rng, B1, data_complement) for B1 in data_B1_array];
            else
                data_B1_perm = [permute(rng, B1, n) for B1 in data_B1_array];
            end

            fill!(testmatrix, 0.0) 
            testmatrix[points_x,points_y] .= 1
            ntests_blocks .+= testmatrix
            
            T0_tmp = stat_test(data_B1,data_B2,data,points_x,points_y)

            fill!(Tperm_tmp, 0.0)

            @showprogress 1 for i in eachindex(Tperm_tmp)
                Tperm_tmp[i] = stat_test(data_B1_perm[i], data_B2, data, points_x, points_y)
            end

            pval_tmp = mean(Tperm_tmp .>= T0_tmp)

            corrected_pval_temp[points_x,points_y] .= pval_tmp
            corrected_pval_temp[points_y,points_x] .= pval_tmp # symetrization

            # old adjusted p-value
            # pval_array[1,:,:] .= corrected_pval 
            # p-value resulting from the test of this block
            # pval_array[2,:,:] .= corrected_pval_temp 
            # maximization for updating the adjusted p-value

            for i in 1:20, j in 1:20
                corrected_pval[i,j] = max(corrected_pval[i,j], corrected_pval_temp[i,j])
            end

            #for i in 1:p, j in 1:p 
            #    index[i,j] = findmax(pval_array[:,i,j])[2] 
            #end

            #responsible_test[ index .== 2] .= "$(paste(index_x))-$(paste(index_y))"
            
        end
    end
    
    corrected_pval
end

iwt_block_precision(data, blocks; B=100)


