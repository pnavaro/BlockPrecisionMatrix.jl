using RCall, Random, GLMNet, InvertedIndices, Statistics, Distributions

nrows, ncols = 8, 5
@rput ncols
@rput nrows
R"set.seed(1234)"
X = rcopy(R"matrix(rnorm(nrows*ncols), nrows, ncols )")
X

k = 1
fitreg = glmnet(X[:,Not(k)],X[:,k], standardize=false)

fitreg.betas



predict(fitreg,X[:,Not(k)])

function prec_xia(X)

    nrows, ncols = size(X)

    betahat = zeros(Float64,(ncols-1,ncols))
    reshat  = copy(X)

    for k in 1:ncols
      fitreg = glmnet(X[:,Not(k)],X[:,k], standardize=false) 
      betahat[:,k] .= vec(fitreg.betas)
      reshat[:,k]  .= X[:,k] .- vec(predict(fitreg,X[:,Not(k)]))
    end

    rtilde = cov(reshat) .* (n-1) ./ n
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
            thetahatij = (1+(betahat[i,j]^2  * rhat[i,i]/rhat[j,j]))/(n*rhat[i,i]*rhat[j,j])   
            TprecStd[i,j] = TprecStd[j,i] = Tprec[i,j]/sqrt(thetahatij)
        end
    end
    
    Tprec, TprecStd
end

prec_xia(X)


