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
    using ParallelDataTransfer
    using CategoricalArrays
end

@sync for w in workers()
   @spawnat w println( " Packages installed" )
end

function run_simulation()

    p = 20 
    n = 1000
    b = 5
    @everywhere rng = MersenneTwister(4272)
    blocs_on  = [(2,4)]
    
    covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

    @passobj 1 workers() data
    @passobj 1 workers() blocks

    display(heatmap(covmat, title="covmat"))
    
    @everywhere begin
        nblocks = length(levels(CategoricalArray(blocks)))
        index_xy = PrecisionMatrix.index_blocks(blocks)
        n_xy = length(index_xy)
    end

    
    # Parallel loop to compute single p-value

    @everywhere function compute_pval( local_pval )
        n, p  = size(data)
        PrecisionMatrix.StatTest(n, p)
        i_xy = localindices(local_pval)
        println(i_xy)
        for k in i_xy
            index_x, index_y = index_xy[k]
            local_pval[k] = PrecisionMatrix.compute_pval(rng, data, stat_test, blocks, index_x, index_y)
        end
    end

    pval = zeros(Float64, (p,p))

    local_pval = SharedVector{Float64}(n_xy, init=compute_pval)

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
