module PrecisionMatrix

using LinearAlgebra
using NCVREG
using Random
using Statistics

include("rotation_matrix.jl")
include("fonctions_simu.jl")
include("prec_xia.jl")
include("stat_test.jl")
include("permute_conditional.jl")
include("precision_iwt.jl")

end
