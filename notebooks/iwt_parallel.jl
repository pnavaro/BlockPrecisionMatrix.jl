using Distributed

import Hwloc

rmprocs(workers())
n = Hwloc.num_physical_cores()
addprocs(n, exeflags=`--project=$@__DIR__`)
@show nprocs()

#-

@show myid()

using UnicodePlots
using Distributed
using CategoricalArrays

@everywhere begin
    using Pkg
    Pkg.instantiate()
    using SharedArrays
    include(joinpath(@__DIR__, "../src/rotation_matrix.jl"))
    include(joinpath(@__DIR__, "../src/structure_cov.jl"))
    include(joinpath(@__DIR__, "../src/cov_simu.jl"))
    include(joinpath(@__DIR__, "../src/generate_data.jl"))
    include(joinpath(@__DIR__, "../src/stat_test.jl"))
    include(joinpath(@__DIR__, "../src/permute_conditional.jl"))
    include(joinpath(@__DIR__, "../src/precision_iwt.jl"))
end

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

    index_xy = index_blocks(blocks)

    n_xy = length(index_xy)

    local_pval = SharedVector{Float64}(n_xy)
    
    # Parallel loop to compute single p-value
    n, p  = size(data)
    stat_test = StatTest(n, p)

    @sync for (i_xy, (index_x, index_y)) in enumerate(index_xy)
    
        @async local_pval[i_xy] = compute_pval(rng, data, stat_test, blocks, index_x, index_y)

    end

    pval = zeros(Float64, (p,p))

    for (i_xy, (index_x, index_y)) in enumerate(index_xy)

        points_x = findall(x -> x in index_x, blocks)
        points_y = findall(x -> x in index_y, blocks)

        for i in points_x, j in points_y
            pval[i,j] = max(pval[i,j], local_pval[i_xy])
            pval[j,i] = pval[i,j]
        end
    
    end
    
    pval

end

@time pval = run_simulation()

display(heatmap(pval, title="pval"))
