# function performing the test on blocks and adjusting the results
# inputs:
# data: data matrix (n*p) where n is the sample size and p the number of grid points (for now. it will be spline coefficients?)
# blocks: vector of length p containing block indexes (numeric, from 1 to number of blocks) for each grid point
# B: number of permutations done to evaluate the tests p-values
IWT_Block_covariance <- function(data,blocks,B=1000){  
  
  n <- dim(data)[1]
  p = dim(data)[2]
  nblocks = length(levels(factor(blocks)))
  
  stat.test = function(block1,block2){
    Rhohat = cor(block1,block2,method='pearson')
    Rohat.std = Rhohat^2/(1-Rhohat^2)
    return(sum(Rohat.std)) # as for now, we use just standardized squared correlation as test statistic
  }
  
  # Permutation test:
  #T_coeff <- array(dim=c(B,p,p)) # permuted test statistics
  permute = function(x,n){
    permutation = sample(n)
    result = x[permutation,]
    return(result) 
  }
  # tests on rectangles
  pval_array = array(dim=c(2,p,p))
  corrected.pval = matrix(0,nrow=p,ncol=p)
  seeds = round(runif(B,0,1000000))
  
  responsible.test = matrix(nrow=p,ncol=p)
  ntests.blocks = zeromatrix = matrix(0,nrow=p,ncol=p)
  
  for(ix in 2:nblocks){ # x coordinate starting point
    for(lx in 0:(nblocks-ix)){ # length on x axis of the rectangle
      index.x = ix:(ix+lx)
      points.x = which(blocks %in% index.x)
      data.B1 = data[,points.x]
      
      data.B1.array = array(data=data.B1,dim=c(n,dim(data.B1)[2],B))
      data.B1.list = lapply(seq(dim(data.B1.array)[3]), function(x) data.B1.array[ , , x])
      data.B1.perm.l = lapply(data.B1.list,permute,n=n)
      data.B1.perm = simplify2array(data.B1.perm.l)
      
      for(iy in 1:(ix-1)){ # y coordinate starting point. stops before the diagonal
        for(ly in 0:(ix-iy-1)){ # length on y axis of the rectangle
          
          index.y = iy:(iy+ly)
          points.y = which(blocks %in% index.y)
          data.B2 = data[,points.y]
          
          testmatrix = zeromatrix
          testmatrix[points.x,points.y] = 1
          ntests.blocks = ntests.blocks + testmatrix
          
          T0.tmp = stat.test(data.B1,data.B2)
          
          Tperm.tmp = apply(data.B1.perm,3,stat.test,block2=data.B2)
          pval.tmp = mean(Tperm.tmp >= T0.tmp)
          
          corrected.pval_temp = matrix(0,nrow=p,ncol=p) 
          corrected.pval_temp[points.x,points.y] = pval.tmp
          corrected.pval_temp[points.y,points.x] = pval.tmp # simmetrization
          pval_array[1,,] = corrected.pval
          pval_array[2,,] = corrected.pval_temp
          
          corrected.pval = apply(pval_array,c(2,3),max)
          
          index = apply(pval_array,c(2,3),which.max)
          responsible.test[which(index==2)] = paste0(paste0(index.x,collapse=','),'-',paste0(index.y,collapse=','))
        }
      }
    }
  }
  return(list(corrected.pval=corrected.pval,responsible.test=responsible.test,ntests.blocks=ntests.blocks))
}

