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
    using BlockPrecisionMatrix
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
    rng = MersenneTwister(4272)
    blocs_on  = [(1,3)]
    
    covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

    display(heatmap(covmat, title="covmat"))
    
    nblocks = length(levels(CategoricalArray(blocks)))
    index_xy = BlockPrecisionMatrix.index_blocks(blocks)
    n_xy = length(index_xy)

    pval = zeros(Float64, (p,p))

    stat_test = BlockPrecisionMatrix.StatTest(n, p)

    bar = Progress(n_xy)

    k = 1
    r = Array{Future}(undef, nworkers())

    for chunk in Iterators.partition( 1:n_xy, nworkers()) 

        for (i,k) in enumerate(chunk)

            w = workers()[i]
            index_x, index_y  = index_xy[k]
            println(" job $k $(first(index_x):last(index_x)) - $(first(index_y):last(index_y)) ")
            r[i] = @spawnat w BlockPrecisionMatrix.compute_pval(rng, data, stat_test, blocks, index_x, index_y)

        end

        for (i,k) in enumerate(chunk)

              local_pval = fetch(r[i])
              index_x, index_y  = index_xy[k]
              points_x = findall(x -> x in index_x, blocks)
              points_y = findall(x -> x in index_y, blocks)

              for i in points_x, j in points_y
                  pval[i,j] = max(pval[i,j], local_pval)
                  pval[j,i] = pval[i,j]
              end
              display(heatmap(pval, title="pval"))

        end

    end
    

    display(heatmap(premat, title="premat"))
    
end

@time run_simulation()
