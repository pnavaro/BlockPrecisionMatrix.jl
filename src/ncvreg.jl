ncol( A::AbstractMatrix ) = size(A)[2]

function standardize(X)
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

end

function std(X)

    STD, center, scale = standardize(X)

    ns = find(STD .> 1e-6)

    if length(ns) == ncol(X)
      val = STD
    else 
      val = STD[:, ns]
    end

    (center=c, scale=s, nonsingular=ns)

end


function ncvreg (X, y, lambda ) 

  family         = "gaussian"
  penalty        = "SCAD"
  gamma          = switch(penalty, SCAD=3.7, 3)
  alpha          = 1 
  lambda.min     = ifelse( n>p, 001, .05)
  nlambda        = 100
  eps            = 1e-4
  max.iter       = 10000
  convex         = true 
  dfmax          = p+1 
  penalty.factor = ones(Float64, size(X)[2])
  warn           = true


  ## Set up XX, yy, lambda
  XX  = std(X)
  p   = size(XX)[2]

  yy  = y .- mean(y)

  n   = length(yy)


  nlambda = length(lambda)
  user.lambda = true
  
  beta, loss, iter = cdfit_gaussian(XX, yy, penalty, lambda, eps, as.integer(max.iter), as.double(gamma), 
      penalty.factor, alpha, as.integer(dfmax), user.lambda )

  a = repeat(mean(y),nlambda)
  b = matrix(beta, p, nlambda)

  ## Eliminate saturated lambda values, if any
  ind     = (iter .> 0)
  a      .= a[ind]
  b      .= b[ :, ind]
  iter   .= iter[ind]
  lambda .= lambda[ind]
  loss   .= loss[ind]

  if (family=="binomial") Eta .= Eta[:,ind] end
  if (warn && sum(iter) == max.iter) @warn "Maximum number of iterations reached"

  ## Local convexity?
  if convex
      convex.min = convexMin(b, XX, penalty, gamma, lambda*(1-alpha), family, penalty.factor, a=a) 
  else 
      convex.min = nothing
  end

  ## Unstandardize
  beta = zeros(Float64, ((ncol(X)+1), length(lambda)))
  bb   = b / attr(XX, "scale")[ns]
  beta[ns+1,] <- bb
  beta[1,] <- a - crossprod(attr(XX, "center")[ns], bb)

  ## Names
  varnames <- if (is.null(colnames(X))) paste("V",1:ncol(X),sep="") else colnames(X)
  varnames <- c("(Intercept)", varnames)
  dimnames(beta) <- list(varnames, lamNames(lambda))

  ## Output
  val <- structure(list(beta = beta,
                        iter = iter,
                        lambda = lambda,
                        penalty = penalty,
                        family = family,
                        gamma = gamma,
                        alpha = alpha,
                        convex.min = convex.min,
                        loss = loss,
                        penalty.factor = penalty.factor,
                        n = n),
                   class = "ncvreg")
  
  
    XX, yy

end
