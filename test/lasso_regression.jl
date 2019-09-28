# -*- coding: utf-8 -*-
# + {}
using Plots, CSV

time = 1:0.1:2
X = reshape(collect(time), size(time)[1], 1)
n = size(X)
y = sin.(2π * collect(time)) #.+ 0.2 *rand(n)
scatter(X, y)

# +
include("../src/lasso.jl")
model = LassoRegression(degree=15, 
                        reg_factor=0.05,
                        learning_rate=0.001,
                        n_iterations=4000)

y_pred = fit_and_predict(model, X, y);

plot(X, y_pred)
scatter!(X, y)
# -
X = reshape(1:5,5,1)


normalize(polynomial_features(X, 3))

X = rand(10,4)

y = collect(1:10)

(y' * X)'

X * y





