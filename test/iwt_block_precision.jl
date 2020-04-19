using CategoricalArrays
using Random, LinearAlgebra, Distributions
using GLMNet
using InvertedIndices
using NCVREG

import PrecisionMatrix: structure_cov
import PrecisionMatrix: cov_simu
import PrecisionMatrix: permute_conditional

p        = 20 
n        = 500
b        = 3 
blocsOn  = [[1,3]]
resBlocs = structure_cov(p, b, blocsOn, seed = 42)
resBlocs[:blocs]

D        = rand(Uniform(1e-4, 1e-2), p)
resmat   = cov_simu(resBlocs[:blocs], resBlocs[:indblocs], 
                   blocsOn, D)

# heatmap(resmat[:CovMat], c=ColorGradient([:red,:blue]))
# heatmap(resmat[:PreMat])

d = MvNormal(resmat[:CovMat])
rng = MersenneTwister(1234)
data = rand!(rng, d , zeros(Float64,(p, n)));

p_part = map( length,  resBlocs[:indblocs])

blocks  =vcat([repeat([i], j) for (i,j) in zip(1:b, p_part)]...)

B = 1000
estimation = :SCAD

n, p  = size(data)

nblocks = length(levels(CategoricalArray(blocks)))

# +
# tests on rectangles
pval_array = zeros(Float64,(2,p,p))
corrected_pval = zeros(Float64, (p,p))
seeds = round.(100000 * rand(B))

responsible_test = zeros(Float64, (p,p))
ntests_blocks = zeros(Float64,(p,p))
zeromatrix = zeros(Float64,(p,p))
B
# -

index = zeros(Int,(p,p))
corrected_pval_temp = zeros(Float64,(p,p));

ix = 2  # for ix in 2:nblocks  # x coordinate starting point.
lx = 0  # for lx in 0:(nblocks-ix) # length on x axis of the rectangle
index_x = ix:(ix+lx) # index first block
points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
data_B1 = data[:,points_x] # data of the first block
data_B1_array = Iterators.repeated(data_B1, B)

data_B1_list = [data_B1 for i in 1:B]

iy = 1 # for iy in 1:(ix-1) # y coordinate starting point. stops before the diagonal
ly = 0 # for ly in 0:(ix-iy-1) # length on y axis of the rectangle

# data of the second block
index_y = iy:(iy+ly) # index second block
points_y = findall( x -> x in index_y, blocks)
data_B2 = data[:,points_y]

index_complement = collect(1:nblocks)[Not([index_x;index_y])]
points_complement = findall( x -> x in index_complement, blocks)
data_complement = data[:,points_complement]

println("x:$index_x")
println("y:$index_y")
println("c:$index_complement")


# permuted data of the first block
ncol = size(data_complement)[2]

if ncol > 0
  data_B1_perm_l = [permute_conditional(B1, n, data_complement, estimation) for B1 in data_B1_list]
else
  data_B1_perm_l = [permute(B1,n) for B1 in data_B1_list]
end

@show data_B1_perm 

#=
                
for (k,m) in enumerate(data_B1_perm_l) 
    data_B1_perm[:,:,k] = m 
end

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

#             end
#         end
#     end
# end

=#
