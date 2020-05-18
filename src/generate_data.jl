using ReadOnlyArrays

export generate_data

"""
    generate_data(rng, p, n, b, blocs_on )

Create the simulation data and returning the covariance matrix and prediction
matrix with blocks.

# Example

```julia
using Random
using UnicodePlots
import PrecisionMatrix.generate_data

p = 10 
n = 500
b = 3
rng = MersenneTwister(12)
blocs_on  = [[1,3]]

covmat, premat, data, blocks = generate_data(rng, p, n, b, blocs_on)

display(heatmap(covmat, title="covmat"))
display(heatmap(premat, title="premat"))
```

"""
function generate_data(rng, p, n, b, blocs_on )

    blocs, indblocs = structure_cov(rng, p, b, blocs_on)

    D = rand(rng, Uniform(1e-4, 1e-2), p)

    covmat, premat = cov_simu(rng, blocs, indblocs, blocs_on, D)
    
    d = MvNormal(covmat)
    p_part = map( length,  indblocs)
    blocks  = vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...)
    data = rand!(rng, d, zeros(Float64,(p, n)))
    covmat, premat, ReadOnlyArray(transpose(data)), blocks

end

