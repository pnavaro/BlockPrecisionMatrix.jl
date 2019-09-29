using Statistics


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

    degree        :: Int64
    reg_factor    :: Float64
    n_iterations  :: Int64
    learning_rate :: Float64
    w             :: Array{Float64}

    function LassoRegression( ; degree, reg_factor, 
                                learning_rate, n_iterations)

        new( degree, reg_factor, n_iterations, learning_rate, Float64[])

    end


    function LassoRegression( X, y; degree, reg_factor, 
                              learning_rate, n_iterations)

        X = normalize(polynomial_features(X, degree))
        # Insert constant ones for bias weights
        X = hcat( ones(eltype(X), size(X)[1]), X)
        n_features = size(X)[2]

        # Initialize weights randomly [-1/N, 1/N]
        limit = 1 / sqrt(n_features)
        w = -limit .+ 2 .* limit .* rand(n_features)

        # Do gradient descent for n_iterations
        for i in 1:n_iterations
            y_pred = X * w
            # Gradient of l2 loss w.r.t w
            grad_w  = - ((y .- y_pred)' * X)' .+ reg_factor .* sign.(w)
            # Update the weights
            w .-= learning_rate .* grad_w
        end

        new( degree, reg_factor, n_iterations, learning_rate, w)

    end

end

function predict(model, X)

    X = normalize(polynomial_features(X, model.degree))
    # Insert constant ones for bias weights
    X = hcat( ones(eltype(X), size(X)[1]), X)
    n_features = size(X)[2]
    return X * model.w

end

function fit_and_predict(model :: LassoRegression, X :: Array{Float64,2}, y :: Vector{Float64})

    X = normalize(polynomial_features(X, model.degree))
    # Insert constant ones for bias weights
    X = hcat( ones(eltype(X), size(X)[1]), X)
    n_features = size(X)[2]

    # Initialize weights randomly [-1/N, 1/N]
    limit = 1 / sqrt(n_features)
    if length(model.w) == 0
        push!(model.w, rand(n_features)...)
    end
    model.w .= -limit .+ 2 .* limit .* model.w

    # Do gradient descent for n_iterations
    for i in 1:model.n_iterations
        y_pred = X * model.w
        # Gradient of l2 loss w.r.t w
        grad_w  = - ((y .- y_pred)' * X)' .+ model.reg_factor .* sign.(model.w)
        # Update the weights
        model.w .-= model.learning_rate .* grad_w
    end

    return X * model.w

end
