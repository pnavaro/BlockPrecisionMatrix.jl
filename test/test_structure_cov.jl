using Random
using BlockPrecisionMatrix
using Distributions

@testset "Simulation functions" begin

p = 10 
n = 500
b = 3
rng = MersenneTwister(12)
blocs_on  = [[1,3]]

blocs, indblocs, matblocs = BlockPrecisionMatrix.structure_cov(rng, p, b, blocs_on)

D = rand(rng, Uniform(1e-4, 1e-2), p)

covmat, premat = BlockPrecisionMatrix.cov_simu(rng, blocs, indblocs, blocs_on, D)

@test true

end
