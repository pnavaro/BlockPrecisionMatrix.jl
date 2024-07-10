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
    using BlockPrecisionMatrix
    using ParallelDataTransfer
end

@sync for w in workers()
   @spawnat w println( " Packages installed" )
end

const p = 20 
const n = 1000
const b = 5
const blocs_on  = [(1,3)]

@everywhere rng = MersenneTwister(4272)

covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

display(heatmap(covmat, title="covmat"))

passobj(1,workers(), [:data, :blocks])

function run_simulation()

    @everywhere begin

        n, p = size(data) 
        stat_test = BlockPrecisionMatrix.StatTest(n, p)
        index_xy = BlockPrecisionMatrix.index_blocks(blocks)

        function compute_pval( i_xy )
            ix, iy = i_xy
            println(" $(first(ix):last(ix)) - $(first(iy):last(iy)) ")
            BlockPrecisionMatrix.compute_pval(rng, data, stat_test, blocks, ix, iy)
        end

    end

    local_pval = pmap(compute_pval, index_xy )

    pval = zeros(Float64, (p,p))

    for k in eachindex(index_xy)

        index_x, index_y = index_xy[k]

        points_x = findall(x -> x in index_x, blocks)
        points_y = findall(x -> x in index_y, blocks)

        for i in points_x, j in points_y
            pval[i,j] = max(pval[i,j], local_pval[k])
            pval[j,i] = pval[i,j]
        end
        display(heatmap(pval, title="pval"))
    
    end

    display(heatmap(premat, title="premat"))
    
end

@time run_simulation()
