using CategoricalArrays
using Random

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
    corrected_pval = zeros(Float64, (p,p))
    Tperm = zeros(Float64, B)

    stat_test = StatTest(n, p)
    permutation = randperm(rng, n)
    
    # x coordinate starting point and length on x axis of the rectangle
    for ix in 2:nblocks, lx in 0:(nblocks-ix)

        # FIRST BLOCK
        index_x = ix:(ix+lx) # index first block
        points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x

        data_B1 = view(data,:,points_x) # data of the first block
        data_B1_perm = copy(data[:, points_x])
        residuals = similar(data_B1_perm)

        # y coordinate starting point. stops before the diagonal and length on y axis of the rectangle
        for iy in 1:(ix-1), ly in 0:(ix-iy-1)

            # SECOND BLOCK

            index_y = iy:(iy+ly) # index second block
            points_y = findall( x -> x in index_y, blocks)

            # data_B2 = view(data,:,points_y) # data of the second block

            # COMPLEMENT

            index_complement = filter(index -> !(index in vcat(index_x,index_y)), 1:nblocks)
            points_complement = findall( x -> x in index_complement, blocks)

            data_complement = view(data, :, points_complement)

            # permuted data of the first block
            ncol = size(data_complement)[2]

            T0 = stat_test(data_B1, data, points_x, points_y)

            ZZ = hcat(ones(n), data_complement)

            for i in eachindex(Tperm)

                permutation .= randperm(rng, n)

                if ncol > 0

                    for j in 1:length(points_x)
                        y = view(data_B1,:,j)
                        λ = 2*sqrt(var(view(data_B1,:,1))*log(length(points_complement))/n)
                        β = NCVREG.coef(NCVREG.SCAD(data_complement, y, [λ]))
                        data_B1_perm[:,j] .= vec(ZZ * β)
                    end
                    
                    residuals .= data_B1 .- data_B1_perm
                    
                    data_B1_perm .+= view(residuals, permutation, :)
                        
                else

                    data_B1_perm .= data[permutation, points_x]

                end

                Tperm[i] = stat_test(data_B1_perm, data, points_x, points_y)

            end

            pval = mean(Tperm .>= T0)

            for i in points_x, j in points_y
                corrected_pval[i,j] = max(corrected_pval[i,j], pval)
                corrected_pval[j,i] = corrected_pval[i,j]
            end

        end
    end
    
    corrected_pval

end
