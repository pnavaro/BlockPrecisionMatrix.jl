using CategoricalArrays
using Distributed
using UnicodePlots
using ProgressMeter

import Hwloc

n = Hwloc.num_physical_cores() - 1
if nprocs() != n+1
    rmprocs(workers())
    addprocs(n, exeflags=`--project`)
end

@show nprocs()

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using SharedArrays
    using Random
    using PrecisionMatrix
end

@sync for w in workers()
   @spawnat w println( " Packages installed" )
end

function run_simulation()

    p = 20 
    n = 1000
    b = 5
    rng = MersenneTwister(4272)
    blocs_on  = [(1,3),(2,4)]
    
    covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)
    
    display(heatmap(covmat, title="covmat"))
    
    nblocks = length(levels(CategoricalArray(blocks)))

    index_xy = PrecisionMatrix.index_blocks(blocks)

    n_xy = length(index_xy)

    
    bar = Progress(n_xy)
    channel = RemoteChannel(()->Channel{Bool}(n_xy), 1)
    pvalues = RemoteChannel(()->Channel{Tuple}(n_xy), 1)
    
    pval = zeros(Float64, (p,p))
    local_pval = SharedVector{Float64}(n_xy)

    @async while take!(channel)

        next!(bar)
        
        k, local_pval = take!(pvalues)
        index_x, index_y = index_xy[k]

        points_x = findall(x -> x in index_x, blocks)
        points_y = findall(x -> x in index_y, blocks)

        for i in points_x, j in points_y
            pval[i,j] = max(pval[i,j], local_pval)
            pval[j,i] = pval[i,j]
        end
        

    end

    # Parallel loop to compute single p-value
    @sync for k in eachindex(index_xy)
    
        index_x, index_y = index_xy[k]

        @spawnat :any begin

            # println(" job $k $(first(index_x):last(index_x)) - $(first(index_y):last(index_y)) ")
            n, p  = size(data)
            stat_test = PrecisionMatrix.StatTest(n, p)
            result = PrecisionMatrix.compute_pval(rng, data, stat_test, blocks, index_x, index_y)
            put!(pvalues, (k, result)) 
            put!(channel, true) 
        end

    end

    put!(channel, false) # this tells the printing task to finish
    close(pvalues)


	display(heatmap(pval, title="pval"))
    display(heatmap(premat, title="premat"))
    
end

@time run_simulation()
