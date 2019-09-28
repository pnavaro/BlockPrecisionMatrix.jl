using Test, LinearAlgebra

@testset "random rotation matrix" begin

    import PrecisionMatrix: RotationMatrix
    
    for p in 1:17
    
        P = RotationMatrix(p)
        @show p
        @test P' * P ≈ diagm(ones(p)) 
        @test P * P' ≈ diagm(ones(p)) 
        @test det(P) ≈ 1.0
    
    end

end
