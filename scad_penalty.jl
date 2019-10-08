# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

using LinearAlgebra, InvertedIndices
using MLDataUtils, ProgressMeter
using Lasso

const MAXIT = 10000
const TOL = 1e-5

function l2_error(th_hat, th_star)
    norm(th_hat .- th_star)
end

function fp(th_hat, th_star)
    fp = 0
    for (t1,t2) in zip(eachrow(th_hat), eachrow(th_star)) 
        if th_star[i] == 0
            if th_hat[i] != 0
                fp += 1
            end
        end
    end
    fp
end

function fn(th_hat, th_star)
    fn = 0
    for i in 1:size(th_star)[1]
        if th_star[i] != 0
            if th_hat[i] == 0
                fn += 1
            end
        end
    end
    fn
end

function evaluate(th_hat, th_star)
    return Dict(:l2_error => l2_error(th_hat, th_star),
            :false_pos => fp(th_hat, th_star),
            :false_neg => fn(th_hat, th_star))
end

function avg_evaluate(evaluates)
    
    avg_l2 = sum(d[:l2_error] for d in evaluates) / length(evaluates)
    avg_fp = sum(d[:false_pos] for d in evaluates) / length(evaluates)
    avg_fn = sum(d[:false_neg] for d in evaluates) / length(evaluates)

    return Dict(:l2_error => avg_l2,
            :false_pos => avg_fp,
            :false_neg => avg_fn)
end

function update_lambda(l, th, a)
    
   for i in eachindex(th)
        if th[i] <= l[i]
            th[i] = l[i]
        elseif l[i] <= th[i] && th[i] <= a*l[i]
            th[i] = (a*l[i] - th[i])/(a-1)
        elseif a*l[i] <= th[i]
            th[i] = 0
        end
    end
    return th
end

function scad(l,th,a)
    for i in eachindex(th)
        if th[i] <= 2*l[i]
            sign(th[i])*(th[i] - l[i])
        elseif 2*l[i] <= th[i] && th[i] <= a*l[i]
            th[i] = sign(th[i])*((a-1)*th[i] - a*l)/(a-2)
        elseif a*l[i] <= th[i]
            th[i] = th[i]
        end
    end
    return th
end

function obj(X, y, th, l)
    1/(2*size(X)[2]) * (norm(y .- X * th)).^2 + scad(l,th)
end

function backtrack(s_prev) end

function lla(X, y, th_0, s_0, l, n)
    i = 0
    a = 3
    th_prev = copy(th_0)
    th_k = similar(th_prev)
    for i in 1:MAXIT
        th_k .= ISTA(X, y, th_prev, s_0, l, n)
        l = update_lambda(l,abs.(th_k),a)
        if norm(th_k .- th_prev,Inf) <= TOL
            break
        end
        th_prev .= th_k
    end
    return th_k, l
end

function ISTA(X, y, th_0, s_0, l, n)
    i = 0
    th_prev = copy(th_0)
    th_k = similar(th_prev)
    s_k = s_0
    for i in 1:MAXIT
        th_k .= th_prev .+ s_k/n * X * (y .- X' * th_prev)
        th_k .= sign.(th_k) .* max.(abs.(th_k) .- s_k*l, zeros(d))
        if norm(th_k .- th_prev,Inf) <= TOL
            break
        end
        th_prev .= th_k
    end

    return th_k
end

function gen_5folds(X, y)
    X_folds = kfolds(X, 5)
    y_folds = kfolds(y, 5)
    return X_folds, y_folds
end

function cv_lambda(X, y, th_0, s_0, lmbda_0, n)
    a = 3
    X_folds, y_folds = gen_5folds(X, y)
    errs = Float64[]
    for (X,y) in zip(X_folds, y_folds)
        th_hat, l = lla(X, y, th_0, s_0, lmbda_0, n)
        push!(errs, 1/(2*size(y))*norm(y .- X * th_hat)^2 + sum(scad(l, abs.(th_hat), a)))
    end
    return sum(errs)/length(X_folds)
end

N = [100, 200]
D = [256, 512, 1024]
s = 10
n = N[1]
d = D[1]
lmbda = 0.85

function run()
    for n in N
        for d in D
            evals_lasso = Float64[]
            evals_ista = Float64[]
            @showprogress 1 for i in 1:200
                cov = 1.0 * Matrix(I,d,d)
                X = rand(MvNormal(cov),n)
                e = rand(Normal(0,1.5),n)
                th_star = vcat([2,2,2,-1.5,-1.5,-1.5,2,2,2,2], zeros(d-10))
                y = X' * th_star .+ e
    
                evs = eigvals(X'X)
                L = maximum(real(evs))
                s_0 = 1/L
                th_0 = zeros(Float64, d)
                lmbda_0 = lmbda * ones(d)
    
                th_hat,l_hat = lla(X, y, th_0, s_0, lmbda_0, n)
                dist = Normal()
                link = IdentityLink()
                X = transpose(X)
                clf = fit(LassoPath,  X, y, dist, link; α = lmbda)
                th_lasso = clf.coefs
                @show size(th_lasso)
                push!(evals_lasso, evaluate(th_lasso, th_star))
                push!(evals_ista, evaluate(th_hat, th_star))
            end
            println("lasso",n,d,avg_evaluate(evals_lasso))
            println("ista",n,d,avg_evaluate(evals_ista))
        end
    end
end

run()


