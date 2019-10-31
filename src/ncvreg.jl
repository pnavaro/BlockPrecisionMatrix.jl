ncol( A::AbstractMatrix ) = size(A)[2]

export mystd

function mystd(X)

    # Declarations
    n, p = size(X)

    XX  = similar(X)
    c   = zeros(Float64, p)
    s   = zeros(Float64, p)

    for j in 1:p

        # Center
        c[j] = 0
        for i in 1:n
          c[j] += X[i, j]
        end
        c[j] = c[j] / n
        for i in 1:n
          XX[i,j] = X[i,j] - c[j]
        end

        # Scale
        s[j] = 0
        for i in 1:n
          s[j] += XX[i,j]^2
        end
        s[j] = sqrt(s[j]/n)
        for i in 1:n
            XX[i,j] = XX[i,j]/s[j]
        end

    end

    XX, c, s

    ns = findall(s .> 1e-6)

    if length(ns) == length(s)
      (values = XX, center=c, scale=s, nonsingular=ns)
    else 
      (values = XX[:,ns], center=c, scale=s, nonsingular=ns)
    end


end


function ncvreg(X, y, lambda ) 

    family         = "gaussian"
    penalty        = "SCAD"
    gamma          = 3.7
    alpha          = 1 
    lambda.min     = ifelse( n>p, 0.001, .05)
    nlambda        = 100
    eps            = 1e-4
    max_iter       = 10000
    convex         = true 
    dfmax          = p+1 
    penalty_factor = ones(Float64, size(X)[2])
    warn           = true

    ## Set up XX, yy, lambda
    XX  = mystd(X)["val"]
    p   = size(XX)[2]

    yy  = y .- mean(y)

    n   = length(yy)

    nlambda = length(lambda)
    user_lambda = true
    
    beta, loss, iter = cdfit_gaussian(XX, yy, penalty, lambda, eps, max_iter, gamma, 
        penalty_factor, alpha, dfmax, user_lambda )

    a = repeat(mean(y),nlambda)
    b = matrix(beta, p, nlambda)

    ## Eliminate saturated lambda values, if any
    ind     = (iter .> 0)
    a      .= a[ind]
    b      .= b[ :, ind]
    iter   .= iter[ind]
    lambda .= lambda[ind]
    loss   .= loss[ind]

    if (warn && sum(iter) == max.iter) @warn "Maximum number of iterations reached" end

    ## Unstandardize
    beta = zeros(Float64, ((ncol(X)+1), length(lambda)))
    bb   = b ./ XX.scale[ns]
    beta[ns+1,] .= bb
    beta[1,:] .= a .- cross(XX.center[ns], bb)

    beta

end
