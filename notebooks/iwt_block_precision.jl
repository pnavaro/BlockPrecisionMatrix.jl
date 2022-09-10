# -*- coding: utf-8 -*-
using CategoricalArrays
using Random
using LinearAlgebra
using Distributions
using GLMNet
using InvertedIndices
using NonConvexPenalizedRegression
using InvertedIndices
using Statistics
using UnicodePlots

include("../src/rotation_matrix.jl")
include("../src/structure_cov.jl")
include("../src/cov_simu.jl")
include("../src/generate_data.jl")
include("../src/stat_test.jl")
include("../src/permute_conditional.jl")

function iwt_block_precision(data, blocks; B=1000)
    
    n, p  = size(data)
    nblocks = length(levels(CategoricalArray(blocks)))
    # tests on rectangles
    pval_array = zeros(Float64,(2,p,p))
    corrected_pval = zeros(Float64, (p,p))
    responsible_test = Array{String,2}(undef,p,p)
    ntests_blocks = zeros(Int,(p,p))
    testmatrix = zeros(Int,(p,p))
    index = zeros(Int,(p,p))
    corrected_pval_temp = zeros(Float64,(p,p))
    Tperm_tmp = zeros(Float64, B)

    stat_test = StatTest(n, p)

    # x coordinate starting point and length on x axis of the rectangle
    for ix in 2:nblocks, lx in 0:(nblocks-ix)

        # FIRST BLOCK

        index_x = ix:(ix+lx) # index first block
        points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
                      

        # y coordinate starting point. stops before the diagonal and length on y axis of the rectangle
        for iy in 1:(ix-1), ly in 0:(ix-iy-1)

            # SECOND BLOCK

            index_y = iy:(iy+ly) # index second block
            points_y = findall( x -> x in index_y, blocks)

            index_complement = filter(index -> !(index in vcat(index_x,index_y)), 1:nblocks)
            points_complement = findall( x -> x in index_complement, blocks)

            data_B1 = view(data,:,points_x)
            data_B2 = view(data,:,points_y)
            data_complement = view(data,:,points_complement)

            # permuted data of the first block
            ncol = size(data_complement)[2]

            fill!(testmatrix, 0) 
            testmatrix[points_x,points_y] .+= 1
            ntests_blocks .+= testmatrix

            display(heatmap(testmatrix,    title="Blocks: $index_x - $index_y"))
            display(heatmap(ntests_blocks, title="Blocks: $index_x - $index_y"))
            
            T0_tmp = stat_test(data_B1, data, points_x, points_y)

            for i in eachindex(Tperm_tmp)

                if ncol > 0
                    data_B1_perm = permute_scad(rng, data_B1, data_complement)
                else
                    data_B1_perm = permute(rng, data_B1, n)
                end

                Tperm_tmp[i] = stat_test(data_B1_perm, data, points_x, points_y)

            end

            pval_tmp = mean(Tperm_tmp .>= T0_tmp)

            corrected_pval_temp[points_x,points_y] .= pval_tmp
            corrected_pval_temp[points_y,points_x] .= pval_tmp # symetrization

            # old adjusted p-value
            # pval_array[1,:,:] .= corrected_pval 
            # p-value resulting from the test of this block
            # pval_array[2,:,:] .= corrected_pval_temp 
            # maximization for updating the adjusted p-value

            for i in 1:p, j in 1:p
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

p = 20 
n = 1000
b = 5
rng = MersenneTwister(12)
blocs_on  = [[1,3]]

covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

display(heatmap(covmat, title="covmat"))
display(heatmap(premat, title="premat"))

@time res = iwt_block_precision(data, blocks; B=1000)

display(heatmap(res))
