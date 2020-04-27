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
        data_B1 = data[:,points_x] # data of the first block
        data_B1_array = Iterators.repeated(data_B1, B)

        # y coordinate starting point. stops before the diagonal and length on y axis of the rectangle
        for iy in 1:(ix-1), ly in 0:(ix-iy-1)

            # SECOND BLOCK

            index_y = iy:(iy+ly) # index second block
            points_y = findall( x -> x in index_y, blocks)
            data_B2 = data[:,points_y]
            index_complement = collect(filter(index -> !(index in vcat(index_x,index_y)), 1:nblocks))
            points_complement = findall( x -> x in index_complement, blocks)
            data_complement = data[:,points_complement]

            # permuted data of the first block
            ncol = size(data_complement)[2]

            if ncol > 0
                data_B1_perm = map( B1 -> permute_scad(rng, B1, data_complement), data_B1_array)
            else
                data_B1_perm = map( B1 -> permute(rng, B1, n), data_B1_array)
            end

            fill!(testmatrix, 0.0) 
            testmatrix[points_x,points_y] .= 1
            ntests_blocks .+= testmatrix

            T0_tmp = stat_test(data_B1,data_B2,data,points_x,points_y)

            Tperm_tmp = map( B1 -> stat_test(B1, data_B2, data, points_x, points_y), data_B1_perm)

            pval_tmp = mean(Tperm_tmp .>= T0_tmp)

            corrected_pval_temp[points_x,points_y] .= pval_tmp
            corrected_pval_temp[points_y,points_x] .= pval_tmp # symetrization

            # old adjusted p-value
            # pval_array[1,:,:] .= corrected_pval 
            # p-value resulting from the test of this block
            # pval_array[2,:,:] .= corrected_pval_temp 
            # maximization for updating the adjusted p-value

            corrected_pval = max.(corrected_pval, corrected_pval_temp)

            #for i in 1:p, j in 1:p 
            #    index[i,j] = findmax(view(pval_array,:,i,j))[2]
            #end

            #responsible_test[ index .== 2] .= "$(paste(index_x))-$(paste(index_y))"
            
        end
    end
    
    corrected_pval

end
