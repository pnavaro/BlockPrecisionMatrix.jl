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

function fit(self, X, y)

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

    return w

end

function predict(self, w, X)

    X = normalize(polynomial_features(X, self.degree))
    # Insert constant ones for bias weights
    X = hcat( ones(eltype(X), size(X)[1]), X)
    n_features = size(X)[2]
    return X * w

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
