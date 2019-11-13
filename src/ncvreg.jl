ncol( A::AbstractMatrix ) = size(A)[2]

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

    any(s .< 1e-6) && throw("Sigular matrix")

    XX, c, s

end


"""
    gaussian_loss( r )

Gaussian loss 
"""
function gaussian_loss( r :: Vector{Float64})::Float64
    l::Float64 = 0
    for i in eachindex(r)
        l = l + r[i]^2
    end
    l
end 

function scad(z::Float64, l1::Float64, l2::Float64, γ::Float64)::Float64
     
    s::Float64 = sign(z)
     
    if abs(z) <= l1 
        return 0
    elseif abs(z) <= (l1*(1+l2)+l1) 
        return s*(abs(z)-l1)/(v*(1+l2))
    elseif abs(z) <= γ*l1*(1+l2)
        return s*(abs(z)-γ*l1/(γ-1))/(1-1/(γ-1)+l2)
    else 
        return z/(1+l2)
    end
    
end

function mcp(z::Float64, l1::Float64, l2::Float64, γ::Float64)::Float64
    
    s :: Float64 = sign(z)
 
    if abs(z) <= l1 
        return 0 
    elseif abs(z) <= γ*l1*(1+l2) 
        return s*(abs(z)-l1)/(1+l2-1/γ)
    else 
        return z/(1+l2)
    end
    
end

function lasso( z::Float64, l1::Float64, l2::Float64)::Float64 

    s::Float64 = sign(z)
    if abs(z) <= l1
       return 0 
    else 
       return s*(abs(z)-l1)/(1+l2)
    end

end

"""
   crossprod( X, y, n, j)

Cross product of y with jth column of X
"""
function crossprod(X, y, j)

    val :: Float64 = 0.0

    for i in eachindex(y)
        val += X[i, j] * y[i];
    end

    return val

end 


"""
    cdfit_gaussian( X, y, penalty, λ, eps, max_iter, γ, multiplier, α, dfmax) 

Coordinate descent for gaussian models
""" 
function cdfit_gaussian( X, y, penalty, λ, eps, max_iter, γ, m, α, dfmax) 

	violations :: Int64 = 0

    n, p = size(X)
    @assert n == length(y)
    L = length(λ)
  
    β = zeros(Float64, (p,L))
    loss = zeros(Float64, L)
    iter = zeros(Int64, L)
  
    a = zeros(Float64, p) # Beta from previous iteration

    tot_iter = 0
  
    r = copy(y)
    z = zeros(Float64, p)
    for j in eachindex(z)
        z[j] = crossprod(X, r, j)/n
    end

    e1 = [false for i in 1:p]
    e2 = [false for i in 1:p]
    
    # If lam[0]=lam_max, skip lam[0] -- closed form sol'n available
    rss = gaussian_loss(r)

    lstart = 1

    sdy = sqrt(rss/n)

    cutoff :: Float64 = 0.0

    # Path
    for l in lstart:L

        if l != 1 

            # Assign a
            for j in eachindex(a)
		       a[j] = β[j,l]
            end

            # Check dfmax
            nv = 0
            for j in eachindex(a)
	            if (a[j] != 0) nv += 1 end
            end
            if ((nv > dfmax) || (tot_iter == max_iter))
                for ll in 1:L inter[l] = -1 end
                break
            end


            # Determine eligible set
            penalty == :lasso  && (cutoff = 2*λ[l] - lam[l-1])
            penalty == :MCP    && (cutoff = λ[l] + γ/(γ-1)*(λ[l] - λ[l-1]))
            penalty == :SCAD   && (cutoff = λ[l] + γ/(γ-2)*(λ[l] - λ[l-1]))

            for j in eachindex(z)
                (abs(z[j]) > (cutoff * α * m[j])) && (e2[j] = 1)
            end

        else 

            # Determine eligible set
            lmax = 0
            for j in eachindex(z) 
                if (abs(z[j]) > lmax) lmax = abs(z[j]) end
            end

            penalty == :lasso && ( cutoff = 2*λ[l] - lmax )
            penalty == :MCP   && ( cutoff = λ[l] + γ/(γ-1)*(λ[l] - lmax) )
            penalty == :SCAD  && ( cutoff = λ[l] + γ/(γ-2)*(λ[l] - lmax) )

            for j in eachindex(z)
                (abs(z[j]) > (cutoff * α * m[j])) && (e2[j] = 1)
            end

        end

        while (tot_iter < max_iter) 
            while (tot_iter < max_iter) 
	            while (tot_iter < max_iter)

	                # Solve over the active set

	                iter[l]  += 1
                    tot_iter +=1 
                    maxChange::Float64 = 0

	                for j in eachindex(z)

	                    if e1[j]

                            z[j] = crossprod(X, r, j) / n + a[j]

	                        # Update β_j
	                        l1 = λ[l] * m[j] * α
	                        l2 = λ[l] * m[j] * (1-α)

	                        penalty == :MCP   && (β[j,l] = mcp(z[j], l1, l2, γ))
	                        penalty == :SCAD  && (β[j,l] = scad(z[j], l1, l2, γ))
                                penalty == :lasso && (β[j,l] = lasso(z[j], l1, l2))

	                        #Update r
	                        shift::Float64 = β[j,l] - a[j]

	                        if shift !=0

                               for i in eachindex(r)
                                   r[i] = r[i] - shift * X[i,j]
                               end

                               (abs(shift) > maxChange) && (maxChange = abs(shift))

                            end
	                 
	                    end

                    end

	                # Check for convergence
	                for j in eachindex(a)
                        a[j] = β[j,l]
                    end

	                (maxChange < eps*sdy) && break

	            end

	            # Scan for violations in strong set
	            violations = 0

	            for j in eachindex(e1)

	                if !e1[j] && e2[j]

	                    z[j] = crossprod(X, r, j) / n

	                    # Update β_j
	                    l1 = λ[l] * m[j] * α
	                    l2 = λ[l] * m[j] * (1-α)

	                    penalty == :MCP    && (β[j,l] = mcp(z[j], l1, l2, γ))
	                    penalty == :SCAD   && (β[j,l] = scad(z[j], l1, l2, γ))
	                    penalty == :lasso  && (β[j,l] = lasso(z[j], l1, l2))

	                    # If something enters the eligible set, update eligible set & residuals
	                    if (β[j,l] !=0)

	                        e1[j] = e2[j] = true
	                        for i in eachindex(r) 
                                r[i] = r[i] - β[j,l]*X[i, j]
                            end
	                        a[j] = β[j,l]
	                        violations += 1

	                    end
	                end 
                end
	    
	            violations==0 && break

            end

            # Scan for violations in rest
            violations = 0

            for j in eachindex(e2)

	            if (e2[j]==0)

	                z[j] = crossprod(X, r, j)/n

	                # Update β_j
	                l1 = λ[l] * m[j] * α
	                l2 = λ[l] * m[j] * (1-α)

	                penalty == :MCP   && (β[j,l] = mcp(z[j], l1, l2, γ))
	                penalty == :SCAD  && (β[j,l] = scad(z[j], l1, l2, γ))
	                penalty == :lasso && (β[j,l] = lasso(z[j], l1, l2))

	                # If something enters the eligible set, update eligible set & residuals

	                if (b[j,l] !=0) 

	                    e1[j] = e2[j] = true
	                    for i in eachindex(r)
                            r[i] -= b[j,l] * X[i, j]
                        end
	                    a[j] = b[j,l]
	                    violations += 1

	                end
                end
            end

            if violations==0 
               break
            end
          
        end

        loss[l] = gaussian_loss(r)
    end 
  
    β, loss, iter

end

export NCVREG

struct NCVREG

    beta :: Array{Float64, 2}

    function NCVREG(X::Array{Float64,2}, y::Vector{Float64}, λ::Vector{Float64} ) 
        penalty        = :SCAD
        γ              = 3.7
        α              = 1 
        eps            = 1e-4
        max_iter       = 10000
        convex         = true 
        n, p           = size(X)
        penalty_factor = ones(Float64, p)
    
        ## Set up XX, yy, λ
        XX, center, scale = mystd(X)
        dfmax = p+1 
    
        yy  = y .- mean(y)
    
        @assert n == length(yy)
    
        λ_min = ifelse( n>p, 0.001, .05)
    
        nλ = length(λ)
        
        β, loss, iter = cdfit_gaussian(XX, yy, penalty, λ, eps, max_iter, γ, 
            penalty_factor, α, dfmax )
    
        ## Unstandardize
        b = β ./ scale
        a = [mean(y) for i in 1:nλ] .- vec(center' * b)
    
        beta = zeros(Float64, (ncol(X)+1, p))
        beta .= transpose(collect(hcat(a, b')))

        new( beta )
    
    end

end

#=
function coef( ncvreg :: NCVREG, lambda, which=1:length(object$lambda), drop=TRUE, ...) {
  if (!missing(lambda)) {
    if (max(lambda) > max(object$lambda) | min(lambda) < min(object$lambda)) {
      stop('Supplied lambda value(s) are outside the range of the model fit.')
    }
    ind <- approx(object$lambda,seq(object$lambda),lambda)$y
    l <- floor(ind)
    r <- ceiling(ind)
    w <- ind %% 1
    beta <- (1-w)*object$beta[,l,drop=FALSE] + w*object$beta[,r,drop=FALSE]
    colnames(beta) <- lamNames(lambda)
  }
  else beta <- object$beta[, which, drop=FALSE]
  if (drop) return(drop(beta)) else return(beta)
end


function predict(ncvreg :: NCVREG, X, type)


=c("link", "response", "class", "coefficients", "vars", "nvars"),
                           lambda, which=1:length(object$lambda), ...) {
  type <- match.arg(type)
  beta <- coef.ncvreg(object, lambda=lambda, which=which, drop=FALSE)
  if (type=="coefficients") return(beta)
  if (class(object)[1]=='ncvreg') {
    alpha <- beta[1,]
    beta <- beta[-1,,drop=FALSE]
  } else {
    beta <- beta
  }

  if (type=="nvars") return(apply(beta!=0,2,sum))
  if (type=="vars") return(drop(apply(beta!=0, 2, FUN=which)))
  eta <- sweep(X %*% beta, 2, alpha, "+")
  if (type=="link" || object$family=="gaussian") return(drop(eta))
  resp <- switch(object$family,
                 binomial = exp(eta)/(1+exp(eta)),
                 poisson = exp(eta))
  if (type=="response") return(drop(resp))
  if (type=="class") {
    if (object$family=="binomial") {
      return(drop(1*(eta>0)))
    } else {
      stop("type='class' can only be used with family='binomial'")
    }
  }
end 
=#
