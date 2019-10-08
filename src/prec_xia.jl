using Random, GLMNet, InvertedIndices
using Statistics, Distributions
using LinearAlgebra

function prec_xia( X :: Array{Float64, 2} )
    
    nrows, ncols = size(X)

    betahat = zeros(Float64,(ncols-1,ncols))
    reshat  = copy(X)
    
    for k in 1:ncols
      y = view(X, :, k )
      λ = [2*sqrt(var(y)*log(ncols)/nrows)]
      fitreg = glmnet(X[:,Not(k)], y, lambda = λ, standardize=false)
      betahat[:,k] .= vec(fitreg.betas)
      reshat[:,k]  .= X[:,k] .- vec(predict(fitreg,X[:,Not(k)]))
    end
    
    rtilde = cov(reshat) .* (nrows-1) ./ nrows
    rhat   = rtilde
    
    for i in 1:ncols-1
       for j in (i+1):ncols
        rhat[i,j] = -(rtilde[i,j]+rtilde[i,i]*betahat[i,j]+rtilde[j,j]*betahat[j-1,i])
        rhat[j,i] = rhat[i,j]
      end
    end
    
    Tprec    = copy(1 ./ rhat)
    TprecStd = copy(Tprec)
    
    for i in 1:(ncols-1)
        for j in (i+1):ncols
            Tprec[i,j] = Tprec[j,i] = rhat[i,j]/(rhat[i,i]*rhat[j,j])
            thetahatij = (1+(betahat[i,j]^2  * rhat[i,i]/rhat[j,j]))/(nrows*rhat[i,i]*rhat[j,j])   
            TprecStd[i,j] = TprecStd[j,i] = Tprec[i,j]/sqrt(thetahatij)
        end
    end

    Tprec, TprecStd
end
