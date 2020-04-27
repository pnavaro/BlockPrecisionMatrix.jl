module PrecisionMatrix

using LinearAlgebra
using Random
using Statistics

include("rotation_matrix.jl")
include("structure_cov.jl")
include("cov_simu.jl")
include("generate_data.jl")
include("stat_test.jl")
include("permute_conditional.jl")
include("precision_iwt.jl")

end
