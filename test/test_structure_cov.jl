using Random
using PrecisionMatrix
using Distributions

@testset "Simulation functions" begin

rng = MersenneTwister(42)
p, n, b  = 20, 500, 3
blocs_on  = [[1,3]]
blocs, indblocs, matblocs = PrecisionMatrix.structure_cov(rng, p, b, blocs_on)

@show blocs
@show indblocs
@show matblocs

D = rand(Uniform(1e-4, 1e-2), p)

covmat, premat = PrecisionMatrix.cov_simu(blocs, indblocs, blocs_on, D)

@test true

end
