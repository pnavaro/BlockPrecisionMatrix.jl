using UnicodePlots
using Random
using BlockPrecisionMatrix
using CategoricalArrays
import Base.Threads: @spawn, threadid, nthreads, @async, @sync

function run_with_threads()

    @show p = 20 
    @show n = 1000
    @show b = 5
    @show blocs_on  = [(1,3)]

    rng = MersenneTwister(4272)
    
    covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

    display(heatmap(covmat, title="covmat"))
    
    nblocks = length(levels(CategoricalArray(blocks)))
    index_xy = BlockPrecisionMatrix.index_blocks(blocks)
    @show n_xy = length(index_xy)

    thread_pval = [zeros(nthreads()) for _ in 1:n_xy]

    @sync for chunk in Iterators.partition( 1:n_xy, n_xy ÷ nthreads()) 

        @async for k in chunk

            @spawn begin 
                w = threadid()
                stat_test = BlockPrecisionMatrix.StatTest(n, p)
                i_x, i_y  = index_xy[k]
                # println("$(threadid()) job $k $(first(i_x):last(i_x)) - $(first(i_y):last(i_y)) ")
                thread_pval[k][w] = BlockPrecisionMatrix.compute_pval(rng, data, stat_test, blocks, i_x, i_y)
            end

        end

    end

    local_pval = maximum.(thread_pval)

    pval = zeros(Float64, (p,p))

    for k in eachindex(index_xy)

        index_x, index_y  = index_xy[k]
        points_x = findall(x -> x in index_x, blocks)
        points_y = findall(x -> x in index_y, blocks)

        for i in points_x, j in points_y
            pval[i,j] = max(pval[i,j], local_pval[k])
            pval[j,i] = pval[i,j]
        end

    end

    
    display(heatmap(pval, title="pval"))

    display(heatmap(premat, title="premat"))
    
end

@time run_with_threads()
