using Distributed
using UnicodePlots

if nworkers() != 4 
   rmprocs(workers()) # remove all worker processes
   addprocs(4); # add worker processes
end

include("../src/rotation_matrix.jl")
include("../src/structure_cov.jl")
include("../src/cov_simu.jl")
include("../src/generate_data.jl")
include("../src/stat_test.jl")
include("../src/permute_conditional.jl")
include("../src/precision_iwt.jl")


p = 20 
n = 1000
b = 3
rng = MersenneTwister(12)
blocs_on  = [[1,3]]

covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

display(heatmap(covmat, title="covmat"))
display(heatmap(premat, title="premat"))

pval = iwt_block_precision(rng, data, blocks; B=1000)

display(heatmap(pval, title="pval"))

const block_tests = RemoteChannel(()->Channel{Int}(32));

const results = RemoteChannel(()->Channel{Tuple}(32));

@everywhere function do_work(jobs, results) 
    while true
        job_id = take!(jobs)
        exec_time = rand()
        sleep(exec_time) # simulates elapsed time doing actual work
        put!(results, (job_id, exec_time, myid()))
    end
end

function make_tests(n)
    for i in 1:n
        put!(block_tests, i)
    end
end

n = 12

@async make_tests(n); # feed the jobs channel with "n" jobs

for p in workers() 
    remote_do(do_work, p, block_tests, results)
end

@elapsed while n > 0 # print out results
    job_id, exec_time, where = take!(results)
    println("$job_id finished in $(round(exec_time; digits=2)) seconds on worker $where")
    global n = n - 1
end
