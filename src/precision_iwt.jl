using CategoricalArrays
using Random

function index_blocks(blocks)

    nblocks = length(levels(CategoricalArray(blocks)))

    index_xy = []

    for ix in 2:nblocks, lx in 0:(nblocks-ix)
        index_x = ix:(ix+lx)
        for iy in 1:(ix-1), ly in 0:(ix-iy-1)
            index_y = iy:(iy+ly)
            push!(index_xy, (index_x, index_y))
        end
    end

    index_xy

end

function compute_pval(rng :: AbstractRNG, data, 
                        stat_test, blocks, index_x, index_y; b = 1000)
    
    n, p = size(data)
    
    points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x
    data_x = view(data,:,points_x) # data of the first block
    data_x_perm = copy(data_x)
    residuals = similar(data_x_perm)
    points_y = findall( x -> x in index_y, blocks)
    nblocks = length(levels(CategoricalArray(blocks)))
    index_z = filter(index -> !(index in vcat(index_x,index_y)), 1:nblocks)
    points_z = findall( x -> x in index_z, blocks)
    data_z = view(data, :, points_z)
    permutation = zeros(Int, n)
    tperm = zeros(b)

    ncol = length(points_z)

    t0 = stat_test(data_x, data, points_x, points_y)

    zz = hcat(ones(n), data_z)

    for i in eachindex(tperm)

        permutation .= randperm(rng, n)

        if ncol > 0

            for j in 1:length(points_x)
                y = view(data_x,:,j)
                λ = 2*sqrt(var(view(data_x,:,1))*log(length(points_z))/n)
                β = NonConvexPenalizedRegression.coef(NonConvexPenalizedRegression.SCAD(data_z, y, [λ]))
                data_x_perm[:,j] .= vec(zz * β)
            end
                
            residuals .= data_x .- data_x_perm
                
            data_x_perm .+= view(residuals, permutation, :)
                    
        else

            data_x_perm .= data[permutation, points_x]

        end

        tperm[i] = stat_test(data_x_perm, data, points_x, points_y)

    end

    return mean(tperm .>= t0)


end

"""
    iwt_block_precision(data, blocks; B=1000, estimation=:SCAD)  

function performing the test on blocks and adjusting the results

- data: data matrix (n*p) where n is the sample size and p the number of grid points (for now. it will be spline coefficients?)
- blocks: vector of length p containing block indexes (numeric, from 1 to number of blocks) for each grid point
- B: number of permutations done to evaluate the tests p-values

"""
function iwt_block_precision(rng :: AbstractRNG, data, blocks; B=1000)
    
    n, p  = size(data)
    nblocks = length(levels(CategoricalArray(blocks)))
    pval = zeros(Float64, (p,p))

    stat_test = StatTest(n, p)

    index_xy = index_blocks(blocks)
    
    for (index_x, index_y) in index_xy

        pvalue = compute_pval(rng, data, stat_test, blocks, index_x, index_y)

        points_x = findall(x -> x in index_x, blocks)
        points_y = findall(x -> x in index_y, blocks)

        for i in points_x, j in points_y
            pval[i,j] = max(pval[i,j], pvalue)
            pval[j,i] = pval[i,j]
        end

    end
    
    pval

end
