using Random, LinearAlgebra, Distributions

# +
include("src/utils.jl")
include("src/rotation_matrix.jl")
include("src/fonctions_simu.jl")
p        = 20 
n        = 500
b        = 3 
blocsOn  = [[1,3]]
resBlocs = structure_cov(p, b, blocsOn, seed = 2)
resBlocs[:blocs]

D        = rand(Uniform(1e-4, 1e-2), p)
resmat   = cov_simu(resBlocs[:blocs], resBlocs[:indblocs], 
                   blocsOn, D)

setdiff(1:b, flatten(blocsOn))
# -


using Plots
heatmap(resmat[:CovMat])

heatmap(resmat[:PreMat])

B1 = 1:16


print(B1...)

length(1:13)






