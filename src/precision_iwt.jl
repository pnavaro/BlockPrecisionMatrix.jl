using Random

"""
    iwt_block_precision(data, blocks; B=1000, estimation=:SCAD)  

function performing the test on blocks and adjusting the results

- data: data matrix (n*p) where n is the sample size and p the number of grid points (for now. it will be spline coefficients?)
- blocks: vector of length p containing block indexes (numeric, from 1 to number of blocks) for each grid point
- B: number of permutations done to evaluate the tests p-values

"""
function iwt_block_precision(data, blocks; B=1000, estimation=:SCAD)  
  
  n, p  = size(data)
  
  nblocks = length(levels(factor(blocks)))
  
  function stat_test(block1_perm, block2, data_orig, points_x, points_y)

    data.orig[:,points.x] .= block1.perm

    PrecMat = precxia(data_orig)

    #Rhohat = cor(block1,block2,method='pearson')
    #Rohat.std = Rhohat^2/(1-Rhohat^2)

    submat = PrecMat.TprecStd[points.x,points.y]

    return(sum(submat)^2) # We use the Xia estimator for the precision matrix

  end
  
  # Permutation test:
  #T_coeff <- array(dim=c(B,p,p)) # permuted test statistics
  function permute(x,n)

      permutation = shuffle(1:n) 

      x[permutation,:] 

  end

#PN  function permute_conditional(y,n,data.complement,estimation) # uses SCAD/OLS
#PN
#PN      permutation = shuffle(1:n)
#PN    
#PN      # SCAD/OLS estimation
#PN      #if estimation == :LM
#PN      #    n = size(data_complement)[2]
#PN      #    fitted = lm.fit(hcat(ones(n),data.complement),y)$fitted,
#PN      #elseif estimation == :SCAD
#PN      lambda = 2 .* sqrt.(var(y[:,1]) .* log.(ncol(data_complement))/nrow(data_complement))
#PN      fitted = [scad_mod(v, data_complement,lambda) for v in eachcol(y)]
#PN      #end
#PN    
#PN      residuals = y .- fitted
#PN    
#PN      result = fitted .+ residuals[permutation,:]
#PN
#PN      return result 
#PN
#PN  end
  
  
  # tests on rectangles
  pval_array = zeros(Float64, (2,p,p))

  corrected_pval = zeros(Float64,(p,p))

  seeds = round(rand(B,0,1000000))
  
  responsible_test = zeros(Float64, (p,p))
  ntests_blocks = zeros(Float64, (p,p))
  
  # For parallelization: the loops are independent between each other, so they can be easily parallelized. 
  # The only point to pay attention to is that for computing the adjusted p-value, the maximization is done at the end of each loop.
  # It would also be possible to save all the results of each loop instead (in an array) and do the maximization only once at the end.
  for ix in 2:nblocks  # x coordinate starting point. 

      for lx in 0:(nblocks-ix) # length on x axis of the rectangle

          index_x = ix:(ix+lx) # index first block
          points_x = find(blocks in index_x) # coefficients in block index.x
          data_B1 = data[:,points_x] # data of the first block
          
          data_B1_array = array(data=data.B1,dim=c(n,dim(data.B1)[2],B))
#PN          #data_B1_list = map(x -> data.B1.array[:,:,x], seq(dim(data.B1.array)[3]))
          
          for iy in 1:(ix-1) # y coordinate starting point. stops before the diagonal

              for ly in 0:(ix-iy-1) # length on y axis of the rectangle

                  # data of the second block
                  index_y = iy:(iy+ly) # index second block
#PN                  points_y = find(blocks %in% index.y)
#PN                  data_B2 = data[:,points_y]
                  
#PN                  index.complement = (1:nblocks)[-c(index.x,index.y)]
#PN                  points.complement = which(blocks %in% index.complement)
#PN                  data.complement = data[,points.complement]
                  
                  # print(paste0('x:',index.x))
                  # print(paste0('y:',index.y))
                  # print(paste0('c:', index.complement))
                  
                  # permuted data of the first block
#PN                  ncol = dim(data.complement)[2]
#PN                  
#PN                  if(ncol > 0){
#PN                    data.B1.perm.l = lapply(data.B1.list,permute.conditional,n=n,data.complement=data.complement,estimation=estimation)
#PN                  }else{
#PN                    data.B1.perm.l = lapply(data.B1.list,permute,n=n)
#PN                  }
#PN                  
#PN                  data.B1.perm = simplify2array(data.B1.perm.l)
#PN
#PN                  testmatrix = zeromatrix
#PN                  testmatrix[points.x,points.y] = 1
#PN                  ntests.blocks = ntests.blocks + testmatrix
#PN                  image.plot(ntests.blocks,main=paste0('Blocks: ',index.x,'-',index.y))
#PN                  image.plot(testmatrix,main=paste0('Blocks: ',index.x,'-',index.y))
#PN
#PN                  T0.tmp = stat.test(data.B1,data.B2,data.orig=data,points.x=points.x,points.y=points.y)
#PN
#PN                  Tperm.tmp = apply(data.B1.perm,3,stat.test,block2=data.B2,data.orig=data,points.x=points.x,points.y=points.y)
#PN                  pval.tmp = mean(Tperm.tmp >= T0.tmp)
#PN
#PN                  corrected.pval_temp = matrix(0,nrow=p,ncol=p)
#PN                  corrected.pval_temp[points.x,points.y] = pval.tmp
#PN                  corrected.pval_temp[points.y,points.x] = pval.tmp # simmetrization
#PN                  
#PN                  pval_array[1,,] = corrected.pval # old adjusted p-value
#PN                  pval_array[2,,] = corrected.pval_temp # p-value resulting from the test of this block
#PN
#PN                  corrected.pval = apply(pval_array,c(2,3),max) # maximization for updating the adjusted p-value
#PN
#PN                  index = apply(pval_array,c(2,3),which.max)
#PN                  responsible.test[which(index==2)] = paste0(paste0(index.x,collapse=','),'-',paste0(index.y,collapse=','))

              end
          end
      end
  end

  (corrected_pval=corrected_pval, responsible_test=responsible_test, ntests_blocks=ntests_blocks)

end

