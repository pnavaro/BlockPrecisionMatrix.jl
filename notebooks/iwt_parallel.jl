using Distributed

import Hwloc

rmprocs(workers())
n = Hwloc.num_physical_cores() - 1
addprocs(n, exeflags=`--project=$@__DIR__`)
@show nprocs()

#-

using UnicodePlots
using Distributed
using CategoricalArrays

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using SharedArrays
    using Random
    using PrecisionMatrix
end

@everywhere println( " Everything is installed on $(myid()) " )

function run_simulation()

    p = 20 
    n = 1000
    b = 5
    rng = MersenneTwister(4272)
    blocs_on  = [[1,3]]
    
    covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)
    
    display(heatmap(covmat, title="covmat"))
    display(heatmap(premat, title="premat"))
    
    nblocks = length(levels(CategoricalArray(blocks)))

    index_xy = PrecisionMatrix.index_blocks(blocks)

    n_xy = length(index_xy)

    local_pval = SharedVector{Float64}(n_xy)
    
    # Parallel loop to compute single p-value

    @sync for (k, (index_x, index_y)) in enumerate(index_xy)
    
        @spawnat :any begin

            println(" job $i $(first(index_x):last(index_x)) - $(first(index_y):last(index_y)) ")
            n, p  = size(data)
            stat_test = PrecisionMatrix.StatTest(n, p)
            rng = MersenneTwister(myid())
            local_pval[k] = PrecisionMatrix.compute_pval(rng, data, stat_test, blocks, index_x, index_y)

        end


    end

    pval = zeros(Float64, (p,p))

    for (k, (index_x, index_y)) in enumerate(index_xy)

        points_x = findall(x -> x in index_x, blocks)
        points_y = findall(x -> x in index_y, blocks)

        for i in points_x, j in points_y
            pval[i,j] = max(pval[i,j], local_pval[k])
            pval[j,i] = pval[i,j]
        end
    
    end
    
    pval

end

@time pval = run_simulation()

display(heatmap(pval, title="pval"))
