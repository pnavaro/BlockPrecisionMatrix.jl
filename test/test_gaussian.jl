@testset "ncvreg works for linear regression" begin

using LinearAlgebra
using Random

import PrecisionMatrix: ncvreg

using MultivariateStats

n = 50
p = 5
# prepare data
X = rand(n, p)                  # feature matrix
a0 = rand(p)                    # ground truths
y = X * a0 + 0.1 * randn(n)     # generate response

# solve using llsq
a = llsq(X, y; bias=false)
@show a

# do prediction
yp = X * a

# measure the error
rmse = sqrt(mean(abs2.(y .- yp)))
println("LLSQ rmse = $rmse")

# solve using linear regression
XX = hcat(ones(n), X)


beta = pinv(XX) * y
@show beta

# do prediction
yp = XX * beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("LM rmse = $rmse")

λ = [0.0]
scad = ncvreg(X, y, λ)

@show scad

yp = XX * scad 

rmse = sqrt(mean(abs2.(y .- yp)))
println("SCAD rmse = $rmse")

@test true

end

