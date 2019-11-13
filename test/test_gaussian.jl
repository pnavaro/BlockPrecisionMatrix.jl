@testset "ncvreg works for linear regression" begin

using LinearAlgebra
using Random
using PrecisionMatrix

Random.seed!(42)

n = 50
p = 5
# prepare data
X = rand(n, p)                  # feature matrix
a0 = rand(p)                    # ground truths
y = X * a0 + 0.1 * randn(n)     # generate response

# solve using linear regression
XX = hcat(ones(n), X)
beta = pinv(XX) * y

# do prediction
yp = XX * beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("LM rmse = $rmse")

λ = [0.0]
scad = NCVREG(X, y, λ)

@test maximum(abs.(beta .- scad.beta)) < 2e-5

yp = XX * scad.beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("SCAD rmse = $rmse")


end
