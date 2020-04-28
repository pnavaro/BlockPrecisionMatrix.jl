import NCVREG

""" 
    permute(rng, x, n) 

Permutation

"""
function permute(rng, x, n) 
   copy(x[randperm(rng, n), :])
end


function permute_lm(rng, y, data_complement)

    n = size(y)[1]

    XX = hcat(ones(n), data_complement)
    beta = pinv(XX) * y
    fitted = XX * beta

    residuals = y .- fitted
    
    permutation = randperm(rng, n)

    fitted .+= view(residuals, permutation, :)
    
    return fitted

end


function permute_scad(rng :: AbstractRNG, Y, Z)

    row_y, col_y = size(Y)
    row_z, col_z = size(Z)
    
    @assert row_y == row_z

    ZZ = hcat(ones(row_z),Z)
    
    fitted = similar(Y)

    for j in 1:col_y
        y = view(Y,:,j)
        λ = 2*sqrt(var(view(Y,:,1))*log(col_z)/row_z)
        β = NCVREG.coef(NCVREG.SCAD(Z, y, [λ]))  
        fitted[:,j] .= vec(ZZ * β)
    end
    
    residuals = Y .- fitted
    
    permutation = randperm(rng, row_y)

    fitted .+= view(residuals, permutation, :)
        
    return fitted
end
