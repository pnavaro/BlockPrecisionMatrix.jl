@testset  "SCAD" begin

    using PrecisionMatrix
    using LinearAlgebra
    using Random
    using NCVREG
    using Statistics
    
    rng = MersenneTwister(123)
    
    n, p = 500, 12
    X = rand(n, p)    
    β = collect(1:p)
    y = X * β + 0.1 * randn(n)
    
    λ = 2*sqrt(var(y) * log(p)/n)

    fitted = PrecisionMatrix.scad_mod( y, X, λ)

    residuals = y .- fitted

    @test sqrt(mean(residuals.^2)) < 5.0


end
