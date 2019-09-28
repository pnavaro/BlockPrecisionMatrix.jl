# -*- coding: utf-8 -*-
# + {}
using Plots

time = 1:0.1:2
X = reshape(collect(time), size(time)[1], 1)
n = size(X)
y = sin.(2π * collect(time)) #.+ 0.2 *rand(n)
scatter(X, y)
# -

IJulia.load("src/lasso.jl")

# +
using Statistics

""" Regularization for Lasso Regression """
struct L1Regularization
    alpha :: Float64
end
    
function(self :: L1Regularization)( w :: Vector{Float64})
    return self.alpha * norm(w)
end

function grad(self :: L1Regularization, w :: Vector{Float64})
    return self.alpha .* sign.(w)
end

"""
    Linear regression model with a regularization factor which does both variable selection 
    and regularization. Model that tries to balance the fit of the model with respect to the training 
    data and the complexity of the model. A large regularization factor with decreases the variance of 
    the model and do para.

    # Parameters:
    
    - `degree`: int
        The degree of the polynomial that the independent variable X will be transformed to.
    - `reg_factor`: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    - `n_iterations`: int
        The number of training iterations the algorithm will tune the weights for.
    - `learning_rate`: float
        The step length that will be used when updating the weights.
"""
struct LassoRegression

    degree         :: Int64
    regularization :: L1Regularization
    n_iterations   :: Int64
    learning_rate  :: Float64

    function LassoRegression( ; degree, reg_factor)

        n_iterations  = 3000
        learning_rate = 0.01
 
        new( degree, L1Regularization(reg_factor), n_iterations, 
             learning_rate)

    end

    function LassoRegression( ; degree, reg_factor, 
                              learning_rate, n_iterations)

        new( degree, L1Regularization(reg_factor), n_iterations, 
             learning_rate)

    end

end

function fit_and_predict(self, X, y)

    X = normalize(polynomial_features(X, self.degree))
    # Insert constant ones for bias weights
    X = hcat( ones(eltype(X), size(X)[1]), X)
    n_features = size(X)[2]

    # Initialize weights randomly [-1/N, 1/N]
    limit = 1 / sqrt(n_features)
    w = -limit .+ 2 .* limit .* rand(n_features)

    # Do gradient descent for n_iterations
    for i in 1:self.n_iterations
        y_pred = X * w
        # Gradient of l2 loss w.r.t w
        grad_w  = - ((y .- y_pred)' * X)' .+ grad(self.regularization, w)
        # Update the weights
        w .-= self.learning_rate .* grad_w
    end

    return X * w

end


# +
import Combinatorics: with_replacement_combinations
using Random, LinearAlgebra


 """ Normalize the dataset X """
function normalize(X)
    l2 = sqrt.(sum(X.^2, dims=2))
    l2 .+= l2 .== 0
    return X ./ l2
end

function polynomial_features(X, degree)

    n_samples, n_features = size(X)

    function index_combinations()
        combs = [with_replacement_combinations(1:n_features, i) for i in 0:degree]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    end
    
    combinations = index_combinations()
    n_output_features = length(combinations)
    X_new = zeros(eltype(X), (n_samples, n_output_features))
    
    for (i, index_combs) in enumerate(combinations)
        X_new[:, i] = prod(X[:, index_combs], dims=2)
    end

    return X_new
end


""" Returns the mean squared error between y_true and y_pred """
function mean_squared_error(y_true, y_pred)
    mse = mean((y_true .- y_pred) .^ 2)
    return mse
end


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





