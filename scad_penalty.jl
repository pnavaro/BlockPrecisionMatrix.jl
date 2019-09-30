# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

using SparseRegression, Lasso, DataFrames
using LinearAlgebra

maxit = 10000
tol = 1e-5

function l2_error(th_hat, th_star)
    norm(th_hat .- th_star)
end

function fp(th_hat, th_star)
    fp = 0
    for i in 1:size(th_star.shape)[1]
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
    for i in 1:size(th_star.shape)[0]
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
    th_prev = th_0
    for i in 1:MAXIT
        th_k = ISTA(X, y, th_prev, s_0, l, n)
        l = update_lambda(l,abs.(th_k),a)
        if norm(th_k .- th_prev,Inf) <= TOL
            break
        end
        th_prev = th_k
    end
    return th_k, l
end

a = [1, 2, 3]
b = [2, 4, 7]
max(a, b)

function ISTA(X, y, th_0, s_0, l, n)
    i = 0
    th_prev = th_0
    s_k = s_0
    for i in 1:MAXIT
        th_k = th_prev .+ s_k/n * X' * (y - X * th_prev)
        th_k = sign.(th_k) * max(abs.(th_k) .- s_k*l, zeros(d))
        if norm(th_k .- th_prev,Inf) <= TOL
            break
        end
        th_prev = th_k
    end

    return th_k
end

function gen_5folds(X, y)
    X_folds = np.split(X, 5)
    y_folds = np.split(y, 5)
    return X_folds, y_folds
end

# +
x = randn(10_000, 50)
y = x * range(-1, stop=1, length=50) + randn(10_000)

s = SModel(x, y, L2DistLoss(), L2Penalty())
@time learn!(s)
s
# -

data = DataFrame(X=[1,2,3], Y=[2,4,7])

m = fit(LassoModel, @formula(Y ~ X), data)

X =[1,2,3]
y = [2, 4, 7]
fit(LassoPath, X, y)


