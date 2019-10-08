using Random, LinearAlgebra

function RotationMatrix(p :: Int64)

    if p == 1 return 1.0*Matrix(I,1,1) end

    P = zeros(Float64, (p,p)) 

    if p == 2
        u = rand()
        sqrt_1_u2 = sqrt(1 - u*u)
        P[1,1] = sqrt_1_u2;    # P[0, 0] = sqrt(1 - u^2) = cos(theta)
        P[2,1] = u;            # P[1, 0] = u = sin(theta)
        P[1,2] = -u;           # P[0, 1] = -u = -sin(theta)
        P[2,2] = sqrt_1_u2;    # P[1, 1] = sqrt(1 - u^2) = cos(theta)
        return P
    end

    Q, R = qr(rand(p, p))
    P = collect(Q)

    if p % 2 == 0

        if first(P) < 0 
           P .*= -1
        end

        E = 1.0 .* Matrix(I, p, p)
        for k in 2:p
            for j in 2:k, i in 1:p
                E[i,j] = P[i,j]
            end
            if det(E) < 0
                P[:,k] .*= -1
            end
        end 
    end
  
    return P

end
