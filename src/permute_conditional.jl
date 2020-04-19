""" 
    permute(x, n) 

Permutation

"""
permute(x :: Array{Float64, 2}, n) = x[randperm(n), :] 



"""
    permute_conditional(y, n, data_complement, estimation)

uses SCAD/OLS

"""
function permute_conditional(y, n, data_complement, estimation)
    # SCAD/OLS estimation
    if estimation == :LM
        XX = hcat(ones(n), data_complement)
        beta = pinv(XX) * y
        # do prediction
        fitted = XX * beta
    elseif estimation == :SCAD
        nrows, ncols = size(data_complement)
        λ = 2*sqrt(var(y[:,1]) * log(ncols)/nrows)
        SCAD = [scad_mod(v, data_complement,λ) for v in eachcol(y)]
    end    

    residuals = y .- fitted
    
    return fitted .+ residuals[permutation,:]
end
