using Lasso
using InvertedIndices

struct StatTest

    n :: Int
    p :: Int
    Tprec :: Array{Float64, 2}
    TprecStd :: Array{Float64, 2}
    β̂ :: Array{Float64, 2}
    reshat :: Array{Float64, 2}
    y :: Vector{Float64}
    r̂ :: Array{Float64, 2}
    r̃ :: Array{Float64, 2}

    function StatTest( n, p)

        Tprec = zeros(p, p)
        TprecStd = zeros(p, p)

        β̂ = zeros(Float64,(p-1,p))
        reshat  = zeros(Float64, (n,p))
        y = zeros(Float64, n)
        d = Normal()
        l = canonicallink(d)
        r̂ = zeros(Float64, (p,p))
        r̃ = zeros(Float64, (p,p))

        new( n, p, Tprec, TprecStd, β̂, reshat, y, r̂, r̃)

    end
    

end

"""
    stat_test(block1_perm, block2, data_orig, points_x, points_y)

We use the Xia estimator for the precision matrix

"""
function (self :: StatTest)(block1_perm, block2, X, points_x, points_y)
    
    X[:, points_x] .= block1_perm

    n, p = size(X)

    d = Normal()
    l = canonicallink(d)
    
    @inbounds for k in 1:p
        x = view(X, :, Not(k))
        for i in eachindex(self.y)
           self.y[i] = X[i, k]
        end
        λ = [2*sqrt(var(self.y)*log(p)/n)]

        fitreg = Lasso.fit(LassoPath, x, self.y, d, l, λ = λ, standardize=false)
        self.y .= vec(Lasso.predict(fitreg, x))
        self.β̂[:,k] .= vec(fitreg.coefs)

        for i in eachindex(self.y)
           self.reshat[i,k] = X[i, k] - self.y[i]
        end

    end
    
    self.r̃ .= cov(self.reshat) .* (n-1) ./ n
    self.r̂ .= self.r̃
    
    @inbounds for i in 1:p-1
        for j in (i+1):p
            self.r̂[i,j] = -(self.r̃[i,j]+self.r̃[i,i]*self.β̂[i,j]+self.r̃[j,j]*self.β̂[j-1,i])
            self.r̂[j,i] = self.r̂[i,j]
        end
    end
    
    @inbounds for i in eachindex(self.Tprec, self.r̂)
        self.Tprec[i] = 1 / self.r̂[i]
    end

    self.TprecStd .= self.Tprec

    @inbounds for i in 1:(p-1)
        for j in (i+1):p
            self.Tprec[i,j] = self.Tprec[j,i] = self.r̂[i,j]/(self.r̂[i,i]*self.r̂[j,j])
            thetahatij = (1+(self.β̂[i,j]^2  * self.r̂[i,i]/self.r̂[j,j]))/(n*self.r̂[i,i]*self.r̂[j,j])   
            self.TprecStd[i,j] = self.TprecStd[j,i] = self.Tprec[i,j]/sqrt(thetahatij)
        end
    end

    submat = view(self.TprecStd,points_x, points_y)
    return sum(submat).^2
    
end
