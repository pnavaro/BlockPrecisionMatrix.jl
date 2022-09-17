@testset "Permutation" begin

    import BlockPrecisionMatrix: permute_scad

    rng = MersenneTwister(111)

    n = 100
    Z = randn(rng, (n,5)) # data complement
    X = randn(rng, (n,3))
    Y = hcat( X * ones(3) .+ 0.1 .* randn(n), Z .+ 0.1 * randn(rng,(n,5)))
    M = hcat(Y,X,Z)
    
    result = permute_scad(rng, Y, Z)
    M1 = hcat(result, X, Z)
    
    @show sum(eigen(inv(cov(M1))).values) # [1] 31.91363
    
    @show sum(eigen(inv(cov(M))).values) # [1] 2346.617
    
    @show sum(var(M,dims=2)) # somme des variances des variables qui composent M
    # [1] 24.05086
    
    @show abs(sum(eigen(inv(cov(M1))).values)-sum(var(M,dims=2)))

    @test true

end
