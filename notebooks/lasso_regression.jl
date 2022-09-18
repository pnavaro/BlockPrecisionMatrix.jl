using BenchmarkTools
using GLMNet, GLM
using InvertedIndices
using Lasso
using LinearAlgebra
using Random
using Test

function test_glmnet(X)
    
    n, p = size(X)

    β = zeros(Float64,(p-1,p))
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
        β[:,k] .= vec(fitreg.betas)
    end

    β
    
end

function test_lasso(X)
    
    n, p = size(X)

    β = zeros(Float64,(p-1,p))
    reshat  = copy(X)
    y = zeros(Float64, n)
    d = Normal()
    l = IdentityLink()
    
    @inbounds for k in 1:p
        x = view(X, :, Not(k))
        for i in eachindex(y)
           y[i] = X[i, k]
        end
        λ = [2*sqrt(var(y)*log(p)/n)]
        lf = Lasso.fit(LassoPath, x, y, d, l, λ = λ, standardize=false)
        y .= vec(Lasso.predict(lf, x))
        β[:,k] .= vec(lf.coefs)
    end

    β  
    
end

rng = MersenneTwister(12)
n = 1000
Z = randn(rng, n, 5)
X = randn(rng, n, 3)
Y = hcat( X * ones(3) .+ 0.1 * randn(rng, n), Z .+ 0.1*randn(rng, n,5))
M = hcat(Y,X,Z)

@btime test_glmnet($M)
@btime test_lasso($M)

norm( test_glmnet(M) .- test_lasso(M))
