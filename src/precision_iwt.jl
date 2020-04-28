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
    
    estimation = :SCAD
    n, p  = size(data)
    nblocks = length(levels(CategoricalArray(blocks)))
    # tests on rectangles
    # pval_array = zeros(Float64,(2,p,p))
    corrected_pval = zeros(Float64, (p,p))
    # responsible_test = Array{String,2}(undef,p,p)
    ntests_blocks = zeros(Float64,(p,p))
    testmatrix = zeros(Float64,(p,p))
    index = zeros(Int,(p,p))
    corrected_pval_temp = zeros(Float64,(p,p))
    Tperm_tmp = zeros(Float64, B)

    stat_test = StatTest(n, p)
    
    # x coordinate starting point and length on x axis of the rectangle
    for ix in 2:nblocks, lx in 0:(nblocks-ix)


        # FIRST BLOCK

        index_x = ix:(ix+lx) # index first block
        points_x = findall(x -> x in index_x, blocks) # coefficients in block index.x

        data_B1 = view(data,:,points_x) # data of the first block


        # y coordinate starting point. stops before the diagonal and length on y axis of the rectangle
        for iy in 1:(ix-1), ly in 0:(ix-iy-1)

            # SECOND BLOCK

            index_y = iy:(iy+ly) # index second block
            points_y = findall( x -> x in index_y, blocks)

            data_B2 = view(data,:,points_y) # data of the second block

            # COMPLEMENT

            index_complement = filter(index -> !(index in vcat(index_x,index_y)), 1:nblocks)
            points_complement = findall( x -> x in index_complement, blocks)

            data_complement = view(data, :, points_complement)

            # permuted data of the first block
            ncol = size(data_complement)[2]

            fill!(testmatrix, 0.0) 
            testmatrix[points_x,points_y] .= 1
            ntests_blocks .+= testmatrix

            T0_tmp = stat_test(data_B1, data, points_x, points_y)

            for i in eachindex(Tperm_tmp)
                if ncol > 0
                    data_B1_perm = permute_scad(rng, data_B1, data_complement)
                else
                    data_B1_perm = permute(rng, data_B1, n)
                end
                Tperm_tmp[i] = stat_test(data_B1_perm, data, points_x, points_y)
            end

            pval_tmp = mean(Tperm_tmp .>= T0_tmp)

            corrected_pval_temp[points_x,points_y] .= pval_tmp
            corrected_pval_temp[points_y,points_x] .= pval_tmp # symetrization

            # old adjusted p-value
            # pval_array[1,:,:] .= corrected_pval 
            # p-value resulting from the test of this block
            # pval_array[2,:,:] .= corrected_pval_temp 
            # maximization for updating the adjusted p-value
            #for i in 1:p, j in 1:p 
            #    index[i,j] = findmax(view(pval_array,:,i,j))[2]
            #end

            #responsible_test[ index .== 2] .= "$(paste(index_x))-$(paste(index_y))"

            corrected_pval = max.(corrected_pval, corrected_pval_temp)
            
        end
    end
    
    corrected_pval

end
