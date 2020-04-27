using GLMNet
using InvertedIndices

"""
    stat_test(block1_perm, block2, data_orig, points_x, points_y)

We use the Xia estimator for the precision matrix

"""
function stat_test(block1_perm, block2, X, points_x, points_y)
    
    X[:, points_x] .= block1_perm

    n, p = size(X)
    Tprec = zeros(p, p)
    TprecStd = zeros(p, p)

    β̂ = zeros(Float64,(p-1,p))
    reshat  = copy(X)
    y = zeros(Float64, n)
    
    @inbounds for k in 1:p
        x = view(X, :, Not(k))
        for i in eachindex(y)
           y[i] = X[i, k]
        end
        λ = [2*sqrt(var(y)*log(p)/n)]
        fitreg = glmnet(x, y, lambda = λ, standardize=false)
        y .= vec(GLMNet.predict(fitreg,x))
        β̂[:,k] .= vec(fitreg.betas)
        for i in eachindex(y)
           reshat[i,k] = X[i, k] - y[i]
        end
    end
    
    r̃ = cov(reshat) .* (n-1) ./ n
    r̂   = r̃
    
    @inbounds for i in 1:p-1
        for j in (i+1):p
            r̂[i,j] = -(r̃[i,j]+r̃[i,i]*β̂[i,j]+r̃[j,j]*β̂[j-1,i])
            r̂[j,i] = r̂[i,j]
        end
    end
    
    @inbounds for i in eachindex(Tprec, r̂)
        Tprec[i] = 1 / r̂[i]
    end

    TprecStd .= Tprec

    @inbounds for i in 1:(p-1)
        for j in (i+1):p
            Tprec[i,j] = Tprec[j,i] = r̂[i,j]/(r̂[i,i]*r̂[j,j])
            thetahatij = (1+(β̂[i,j]^2  * r̂[i,i]/r̂[j,j]))/(n*r̂[i,i]*r̂[j,j])   
            TprecStd[i,j] = TprecStd[j,i] = Tprec[i,j]/sqrt(thetahatij)
        end
    end

    submat = TprecStd[points_x, points_y]
    return sum(submat).^2
    
end
