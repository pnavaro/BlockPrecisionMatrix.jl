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
