using Random, LinearAlgebra, Distributions

# +
include("src/utils.jl")
include("src/rotation_matrix.jl")
include("src/fonctions_simu.jl")
p        = 20 
n        = 500
b        = 3 
blocsOn  = [[1,3]]
resBlocs = structure_cov(p, b, blocsOn, seed = 42)
resBlocs[:blocs]

D        = rand(Uniform(1e-4, 1e-2), p)
resmat   = cov_simu(resBlocs[:blocs], resBlocs[:indblocs], 
                   blocsOn, D)
# -


using Plots
heatmap(resmat[:CovMat])

heatmap(resmat[:PreMat])

datas = rand!( MvNormal(resmat[:CovMat]), zeros(Float64,(p, n)))


indblocs

p_part = map( length,  resBlocs[:indblocs])


blocks  =vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...)


