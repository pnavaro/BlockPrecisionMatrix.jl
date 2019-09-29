# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
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

# # Algorithm
#
# $$
# \mathbf{x}^{(0)} = (x_1^0,...,x_n^0)
# $$
#
# $$
# \mathbf{x}_i^{(k+1)} = argmin_{\omega} f(x_1^{(k+1)}, ..., x_{i-1}^{(k+1)}, \omega, x_{i+1}^{(k)}, ..., x_n^{(k)})
# $$
#
# $$
# x_i : = x_i - \alpha \frac{\partial f}{\partial x_i}(\mathbf{x})
# $$
#
# $$
# \rho_j = \sum_{i=1}^m x_j^{(i)}  (y^{(i)}  - \sum_{k \neq j}^n \theta_k x_k^{(i)} ) = \sum_{i=1}^m x_j^{(i)}  (y^{(i)}  - \hat y^{(i)}_{pred} + \theta_j x_j^{(i)} )
# $$

#Load the diabetes dataset
using CSV
diabetes = CSV.read("diabetes.csv")
X = convert(Matrix,diabetes[:,1:end-1])
y = convert(Vector, diabetes[:,:Outcome])
size(X)

include("src/lasso.jl")

# +
using Plots
# Initialize variables
m,n = size(X)
initial_theta = ones(Float64,n)
theta_lasso = Vector{Float64}[]
lamda = exp10.(range(0, stop=4, length=300)) ./10 #Range of lambda values

#Run lasso regression for each lambda
for l in lamda
    theta = coordinate_descent_lasso(initial_theta, X, y, lamda = l, num_iters=100)
    push!(theta_lasso, copy(theta))
end
theta_lasso = vcat(theta_lasso'...); 
# -

#Plot results
n = length(theta_lasso)
plot(figsize = (12,8))
plot!(theta_lasso, xscale=:log10)

