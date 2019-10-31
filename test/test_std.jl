@testset "std standardizes correctly" begin

import PrecisionMatrix: mystd

rng = MersenneTwister(1234)
n = 20
p = 5
l = 5

X = randn(rng,(n,p))
XX = mystd(X)
@test all(mean(XX.values, dims=1)  .< fill(1e-13, p))
@test all([ c' * c for c in eachcol(XX.values)] .- fill(n, p) .< 1e-13 )

end

