# -*- coding: utf-8 -*-
using Pkg
pkg"add https://github.com/pnavaro/NCVREG.jl"

using CategoricalArrays
using Random, LinearAlgebra, Distributions
using GLMNet
using InvertedIndices
using NCVREG
using Plots

include("../src/rotation_matrix.jl")
include("../src/fonctions_simu.jl")
include("../src/permute_conditional.jl")
include("../src/stat_test.jl")

p         = 20 
n         = 500
b         = 3 
blocs_on  = [[1,3]]
rng = MersenneTwister(42)
blocs, indblocs = structure_cov(rng, p, b, blocs_on)
blocs, indblocs

indblocs

D = rand(rng, Uniform(1e-4, 1e-2), p)
covmat, premat   = cov_simu(blocs, indblocs, blocs_on, D)

hm = plot(layout=(1,2))
heatmap!(hm[1,1], covmat, c=ColorGradient([:red,:blue]), aspect_ratio=:equal)
heatmap!(hm[1,2], premat, aspect_ratio=:equal)

d = MvNormal(covmat)

data = rand!(rng, d, zeros(Float64,(p, n)));

data

p_part = map( length,  indblocs)

blocks  = vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...)

B = 1000
estimation = :SCAD

p, n  = size(data)

nblocks = length(levels(CategoricalArray(blocks)))

# +
# tests on rectangles
pval_array = zeros(Float64,(2,p,p))
corrected_pval = zeros(Float64, (p,p))
seeds = round.(100000 * rand(rng, B))

responsible_test = zeros(Float64, (p,p))
ntests_blocks = zeros(Float64,(p,p))
zeromatrix = zeros(Float64,(p,p))
B
# -

index = zeros(Int,(p,p))
corrected_pval_temp = zeros(Float64,(p,p));

ix = 2  # for ix in 2:nblocks  # x coordinate starting point.
lx = 0  # for lx in 0:(nblocks-ix) # length on x axis of the rectangle

# FIRST BLOCK

index_x = ix:(ix+lx) # index first block
points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
data_B1 = data[:,points_x] # data of the first block
data_B1_array = Iterators.repeated(data_B1, B)

iy = 1 # for iy in 1:(ix-1) # y coordinate starting point. stops before the diagonal
ly = 0 # for ly in 0:(ix-iy-1) # length on y axis of the rectangle

# # SECOND BLOCK block

index_y = iy:(iy+ly) # index second block
points_y = findall( x -> x in index_y, blocks)
data_B2 = data[:,points_y]

index_complement = collect(1:nblocks)[Not([index_x index_y])]
points_complement = findall( x -> x in index_complement, blocks)
data_complement = data[:,points_complement]

println("x:$index_x")
println("y:$index_y")
println("c:$index_complement")

# permuted data of the first block
@show ncol = size(data_complement)[2]

if ncol > 0
    data_B1_perm = [permute_conditional(rng, B1, data_complement) for B1 in data_B1_array]
else
    data_B1_perm = [permute(B1,n) for B1 in data_B1_list]
end

# +
testmatrix = zeromatrix
testmatrix[points_x,points_y] = 1
ntests_blocks = ntests_blocks + testmatrix

heatmap(ntests_blocks, title="Blocks: $index_x - $index_y")
heatmap(testmatrix,    title="Blocks: $index_x - $index_y")

T0_tmp = stat_test(data_B1,data_B2,data,points_x,points_y)

Tperm_tmp = []
for k in 1:size(data_B1_perm)[3]
    push!(Tperm_tmp,stat_test(data_B1_perm[:,:,k],data_B2, data, points_x, points_y))
end

pval_tmp = mean(Tperm_tmp .>= T0_tmp)

corrected_pval_temp[points_x,points_y] = pval_tmp
corrected_pval_temp[points_y,points_x] = pval_tmp # symmetrization

# old adjusted p-value
pval_array[1,:,:] .= corrected_pval 
# p-value resulting from the test of this block
pval_array[2,:,:] .= corrected_pval_temp 

# maximization for updating the adjusted p-value
corrected_pval = maximum(pval_array,dims=1) 

for i in 1:p, j in 1:p 
    index[i,j] = findmax(A[:,i,j])[2] 
end
responsible_test[which(index==2)] = "$(index_x) - $(index_y)"

            end
        end
    end
end

=#
# -


