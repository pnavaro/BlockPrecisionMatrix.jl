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

    local_pval = SharedVector{Float64}(n_xy)
    
    bar = Progress(n_xy)
    channel = RemoteChannel(()->Channel{Bool}(1), 1)
    
    # Parallel loop to compute single p-value
    @async while take!(channel)
       next!(bar)
    end

    @sync for (k, (index_x, index_y)) in enumerate(index_xy)
    
        @spawnat :any begin

            # println(" job $k $(first(index_x):last(index_x)) - $(first(index_y):last(index_y)) ")
            n, p  = size(data)
            stat_test = PrecisionMatrix.StatTest(n, p)
            local_pval[k] = PrecisionMatrix.compute_pval(rng, data, stat_test, blocks, index_x, index_y)
            put!(channel, true) 
        end

    end

    put!(channel, false) # this tells the printing task to finish

    pval = zeros(Float64, (p,p))

    for (k, (index_x, index_y)) in enumerate(index_xy)

        points_x = findall(x -> x in index_x, blocks)
        points_y = findall(x -> x in index_y, blocks)

        for i in points_x, j in points_y
            pval[i,j] = max(pval[i,j], local_pval[k])
            pval[j,i] = pval[i,j]
        end
    
    end

    display(heatmap(premat, title="premat"))
    display(heatmap(pval, title="pval"))
    
end

@time run_simulation()
