using Distributed
using UnicodePlots

if nworkers() != 4 
   rmprocs(workers()) # remove all worker processes
   addprocs(4); # add worker processes
end

@everywhere begin

    using SharedArrays
    using CategoricalArrays
    
    include(joinpath(@__DIR__, "..", "src", "rotation_matrix.jl"))
    include(joinpath(@__DIR__, "..", "src", "structure_cov.jl"))
    include(joinpath(@__DIR__, "..", "src", "cov_simu.jl"))
    include(joinpath(@__DIR__, "..", "src", "generate_data.jl"))
    include(joinpath(@__DIR__, "..", "src", "stat_test.jl"))
    include(joinpath(@__DIR__, "..", "src", "permute_conditional.jl"))
    include(joinpath(@__DIR__, "..", "src", "precision_iwt.jl"))

end

p = 20 
n = 1000
b = 3
rng = MersenneTwister(12)
blocs_on  = [[1,3]]

covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

display(heatmap(covmat, title="covmat"))
display(heatmap(premat, title="premat"))

const block_tests = RemoteChannel(()->Channel{Tuple}(32));
const results = RemoteChannel(()->Channel{Tuple}(32));

@everywhere function do_work(jobs, results) 

    while true
        blocks, ix, iy = take!(jobs)
        px = findall(x -> x in ix, blocks) # coefficients in block index.x
        py = findall(x -> x in iy, blocks)
        nblocks = length(levels(CategoricalArray(blocks)))
        iz = filter(i -> !(i in vcat(ix,iy)), 1:nblocks)
        pz = findall( x -> x in iz, blocks)

        # rng = MersenneTwister(myid())


        put!(results, (px, py, pz))
    end
end

index_xy = index_blocks(blocks)
n = length(index_xy)
println( " Number of tasks $n")

function make_tests(index_xy, blocks)
    for (ix, iy) in index_xy
        put!(block_tests, (blocks, ix, iy))
    end
end

@async make_tests(index_xy, blocks)

for p in workers() 
    remote_do(do_work, p, block_tests, results)
end

while n > 0 # print out results
    px, py, pz = take!(results)
    print("px : $(first(px))-$(last(px)) \t")
    print("py : $(first(py))-$(last(py)) \t")
    if length(pz) > 0
        println("pz : $(first(pz))-$(last(pz))")
    else
        println("pz : 0-0")
    end
    global n = n - 1
end


# futures = Array{Future}(undef, nworkers())
# @time begin
# 
#     for (ix, iy) in index_xy
#         n, p = size(data)
#         stat_test = StatTest(n, p)
#         futures[i] = @spawnat id compute_pval!(pval, rng, data, stat_test, blocks, ix, iy)
#     end
#     p = max(fetch.(futures))
# end
# 
# println(p)
