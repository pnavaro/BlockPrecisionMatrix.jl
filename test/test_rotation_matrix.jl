@testset "random rotation matrix" begin

    import PrecisionMatrix: rotation_matrix
    using Random

    rng = MersenneTwister(123)
    
    for p in 1:17
    
        P = rotation_matrix(rng, p)
        @test P' * P  ≈ diagm(ones(p)) 
        @test P  * P' ≈ diagm(ones(p)) 
        @test det(P)  ≈ 1.0
    
    end

end
