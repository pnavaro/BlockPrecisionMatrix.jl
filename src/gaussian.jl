
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
     
    if (abs(z) <= l1) 
        return 0
    elseif (abs(z) <= (l1*(1+l2)+l1)) 
        return (s*(abs(z)-l1)/(v*(1+l2)))
    elseif (abs(z) <= gamma*l1*(1+l2)) 
        return (s*(abs(z)-gamma*l1/(gamma-1))/(v*(1-1/(gamma-1)+l2)))
    else 
        return (z/(v*(1+l2)))
    end
    
end

function MCP(z::Float64, l1::Float64, l2::Float64, gamma::Float64, v::Float64)::Float64
    
    s :: Float64 = sign(z)
 
    if (abs(z) <= l1) 
        return 0 
    elseif (abs(z) <= gamma*l1*(1+l2)) 
        return (s*(abs(z)-l1)/(v*(1+l2-1/gamma))) 
    else 
        return(z/(v*(1+l2)))
    end
    
end

function lasso(double z, double l1, double l2, double v)::Float64 

    s::Float64 = sign(z)
    if (abs(z) <= l1) return 0 
    else 
       return(s*(fabs(z)-l1)/(v*(1+l2)));
    end

end


"""
Coordinate descent for gaussian models
""" 
function cdfit_gaussian( X, y, penalty, lambda, eps, max_iter, gamma, multiplier, alpha, dfmax, user) 


    n = length(y)
    p = length(X)/n
    L = length(lambda)
  
    beta = zeros(Float64, L*p)
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
		       a[j] = b[(l-1)*p+j]
            end

            # Check dfmax
            nv = 0
            for j in eachinex(a)
	            (a[j] != 0) && nv += 1
            end
            if ((nv > dfmax) || (tot_iter == max_iter))
                for ll in 1:L inter[l] = -1 end
                break
            end

            # Determine eligible set
            penalty == "lasso" && cutoff = 2*lam[l] - lam[l-1]
            penalty == "MCP"   && cutoff = lam[l] + gamma/(gamma-1)*(lam[l] - lam[l-1])
            penalty == "SCAD"  && cutoff = lam[l] + gamma/(gamma-2)*(lam[l] - lam[l-1])

            for j in eachindex(z)
                (abs(z[j]) > (cutoff * alpha * m[j])) && e2[j] = 1
            end

        else 

            # Determine eligible set
            lmax = 0
            for j in eachindex(z) 
                (abs(z[j]) > lmax) && lmax = abs(z[j])
            end

            penalty == "lasso" && cutoff = 2*lam[l] - lmax;
            penalty == "MCP"   && cutoff = lam[l] + gamma/(gamma-1)*(lam[l] - lmax)
            penalty == "SCAD"  && cutoff = lam[l] + gamma/(gamma-2)*(lam[l] - lmax)

            for j in eachindex(z)
                (abs(z[j]) > (cutoff * alpha * m[j])) && e2[j] = 1
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
	                        penalty == "MCP"   && b[l*p+j] = MCP(z[j], l1, l2, gamma, 1)
	                        penalty == "SCAD"  && b[l*p+j] = SCAD(z[j], l1, l2, gamma, 1)
                            penalty == "lasso" && b[l*p+j] = lasso(z[j], l1, l2, 1)

	                        #Update r
	                        shift::Float64 = b[l*p+j] - a[j]
	                        if (shift !=0) 

                               for i in eachindex(r)
                                   r[i] = r[i] - shift*X[j*n+i]
                               end

                               (abs(shift) > maxChange) && maxChange = abs(shift)

                            end
	                 
	                    end

                    end

	                # Check for convergence
	                for j in eachindex(a)
                        a[j] = b[l*p+j]
                    end

	                if (maxChange < eps*sdy) break end

	            end

	            # Scan for violations in strong set
	            violations :: Int64 = 0

	            for j in eachindex(e1)

	                if (e1[j]==0 && e2[j]==1)

	                    z[j] = crossprod(X, r, n, j)/n;

	                    # Update beta_j
	                    l1 = lam[l] * m[j] * alpha;
	                    l2 = lam[l] * m[j] * (1-alpha);

	                    penalty == "MCP"    && b[l*p+j] = MCP(z[j], l1, l2, gamma, 1)
	                    penalty == "SCAD"   && b[l*p+j] = SCAD(z[j], l1, l2, gamma, 1)
	                    penalty == "lasso"  && b[l*p+j] = lasso(z[j], l1, l2, 1)

	                    # If something enters the eligible set, update eligible set & residuals
	                    if (b[l*p+j] !=0)

	                        e1[j] = e2[j] = 1
	                        for i in eachindex(r) 
                                r[i] = r[i] - b[l*p+j]*X[j*n+i];
                            end
	                        a[j] = b[l*p+j];
	                        violations += 1

	                    end
	                end 
                end
	    
	            if (violations==0) break end
            end

            # Scan for violations in rest
            violations :: Int64 = 0;

            for j in eachindex(e2)

	            if (e2[j]==0)

	                z[j] = crossprod(X, r, n, j)/n;

	                # Update beta_j
	                l1 = lam[l] * m[j] * alpha;
	                l2 = lam[l] * m[j] * (1-alpha);

	                penalty == "MCP"    && b[l*p+j] = MCP(z[j], l1, l2, gamma, 1);
	                penalty == "SCAD"   && b[l*p+j] = SCAD(z[j], l1, l2, gamma, 1);
	                penalty == "lasso"  && b[l*p+j] = lasso(z[j], l1, l2, 1);

	                # If something enters the eligible set, update eligible set & residuals

	                if (b[l*p+j] !=0) 

	                    e1[j] = e2[j] = 1;
	                    for (int i=0; i<n; i++) r[i] -= b[l*p+j]*X[j*n+i];
	                    a[j] = b[l*p+j];
	                    violations++;

	                end
                end
            end

            if (violations==0) break end
          
        end

        loss[l] = gaussian_loss(r, n)
    end 
  
    beta, loss, iter

end


