
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

function SCAD(z::Float64, l1::Float64, l2::Float64, gamma::Float64, v::Float64)::Float64
     
    s::Float64 = sign(z)
     
    if abs(z) <= l1 
        return 0
    elseif abs(z) <= (l1*(1+l2)+l1) 
        return s*(abs(z)-l1)/(v*(1+l2))
    elseif abs(z) <= gamma*l1*(1+l2)
        return s*(abs(z)-gamma*l1/(gamma-1))/(v*(1-1/(gamma-1)+l2))
    else 
        return z/(v*(1+l2))
    end
    
end

function MCP(z::Float64, l1::Float64, l2::Float64, gamma::Float64, v::Float64)::Float64
    
    s :: Float64 = sign(z)
 
    if abs(z) <= l1 
        return 0 
    elseif abs(z) <= gamma*l1*(1+l2) 
        return s*(abs(z)-l1)/(v*(1+l2-1/gamma))
    else 
        return z/(v*(1+l2))
    end
    
end

function lasso( z::Float64, l1::Float64, l2::Float64, v::Float64)::Float64 

    s::Float64 = sign(z)
    if abs(z) <= l1
       return 0 
    else 
       return s*(abs(z)-l1)/(v*(1+l2))
    end

end


"""
Coordinate descent for gaussian models
""" 
function cdfit_gaussian( X, y, penalty, lambda, eps, max_iter, gamma, multiplier, alpha, 
    dfmax, user) 

	violations :: Int64 = 0

    n = length(y)
    p = length(X)/n
    L = length(lambda)
  
    beta = zeros(Float64, (L,p))
    loss = zeros(Float64, L)
    iter = zeros(Int64, L)
  
    a = zeros(Float64, p) # Beta from previous iteration

    tot_iter = 0
  
    r = copy(y)
    z = zeros(Float64, p)
    for j in eachinedx(z)
        z[j] = (view(X,:,j) × r)/n
    end

    e1 = zeros(Int64, p)
    e2 = zeros(Int64, p)
    
    # If lam[0]=lam_max, skip lam[0] -- closed form sol'n available
    rss = gLoss(r)
    if (user)
        lstart = 0
    else 
        loss = rss
        lstart = 1
    end

    sdy = sqrt(rss/n)

    # Path
    for l in lstart:L

        if l != 0 

            # Assign a
            for j in eachindex(a)
		       a[j] = b[j,l-1]
            end

            # Check dfmax
            nv = 0
            for j in eachinex(a)
	            if (a[j] != 0) nv += 1 end
            end
            if ((nv > dfmax) || (tot_iter == max_iter))
                for ll in 1:L inter[l] = -1 end
                break
            end

            # Determine eligible set
            if penalty == :lasso  cutoff = 2*lam[l] - lam[l-1] end
            if penalty == :MCP    cutoff = lam[l] + gamma/(gamma-1)*(lam[l] - lam[l-1]) end
            if penalty == :SCAD   cutoff = lam[l] + gamma/(gamma-2)*(lam[l] - lam[l-1]) end

            for j in eachindex(z)
                if (abs(z[j]) > (cutoff * alpha * m[j])) e2[j] = 1 end
            end

        else 

            # Determine eligible set
            lmax = 0
            for j in eachindex(z) 
                if (abs(z[j]) > lmax) lmax = abs(z[j]) end
            end

            if penalty == :lasso  cutoff = 2*lam[l] - lmax end
            if penalty == :MCP    cutoff = lam[l] + gamma/(gamma-1)*(lam[l] - lmax) end
            if penalty == :SCAD   cutoff = lam[l] + gamma/(gamma-2)*(lam[l] - lmax) end

            for j in eachindex(z)
                if (abs(z[j]) > (cutoff * alpha * m[j])) e2[j] = 1 end
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

                            z[j] = cross(view(X,:,j), r) / n + a[j]

	                        # Update beta_j
	                        l1 = lam[l] * m[j] * alpha
	                        l2 = lam[l] * m[j] * (1-alpha)
	                        if penalty == :MCP   b[j,l] = MCP(z[j], l1, l2, gamma, 1) end
	                        if penalty == :SCAD  b[j,l] = SCAD(z[j], l1, l2, gamma, 1) end
                            if penalty == :lasso b[j,l] = lasso(z[j], l1, l2, 1) end

	                        #Update r
	                        shift::Float64 = b[j,l] - a[j]
	                        if (shift !=0) 

                               for i in eachindex(r)
                                   r[i] = r[i] - shift*X[i,j]
                               end

                               if (abs(shift) > maxChange) maxChange = abs(shift) end

                            end
	                 
	                    end

                    end

	                # Check for convergence
	                for j in eachindex(a)
                        a[j] = b[j,l]
                    end

	                if (maxChange < eps*sdy) break end

	            end

	            # Scan for violations in strong set
	            violations = 0

	            for j in eachindex(e1)

	                if (e1[j]==0 && e2[j]==1)

	                    z[j] = crossprod(X, r, n, j)/n;

	                    # Update beta_j
	                    l1 = lam[l] * m[j] * alpha;
	                    l2 = lam[l] * m[j] * (1-alpha);

	                    if penalty == :MCP    b[j,l] = MCP(z[j], l1, l2, gamma, 1) end
	                    if penalty == :SCAD   b[j,l] = SCAD(z[j], l1, l2, gamma, 1) end
	                    if penalty == :lasso  b[j,l] = lasso(z[j], l1, l2, 1) end

	                    # If something enters the eligible set, update eligible set & residuals
	                    if (b[l*p+j] !=0)

	                        e1[j] = e2[j] = 1
	                        for i in eachindex(r) 
                                r[i] = r[i] - b[j, l]*X[i, j]
                            end
	                        a[j] = b[j,l]
	                        violations += 1

	                    end
	                end 
                end
	    
	            if (violations==0) break end
            end

            # Scan for violations in rest
            violations  = 0

            for j in eachindex(e2)

	            if (e2[j]==0)

	                z[j] = crossprod(X, r, n, j)/n;

	                # Update beta_j
	                l1 = lam[l] * m[j] * alpha;
	                l2 = lam[l] * m[j] * (1-alpha);

	                if penalty == :MCP    b[j,l] = MCP(z[j], l1, l2, gamma, 1) end
	                if penalty == :SCAD   b[j,l] = SCAD(z[j], l1, l2, gamma, 1) end
	                if penalty == :lasso  b[j,l] = lasso(z[j], l1, l2, 1) end

	                # If something enters the eligible set, update eligible set & residuals

	                if (b[j,l] !=0) 

	                    e1[j] = e2[j] = 1
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

        loss[l] = gaussian_loss(r, n)
    end 
  
    beta, loss, iter

end


