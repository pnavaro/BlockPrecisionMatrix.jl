
blocks_loop_unadjusted=function(ix,nblocks,data,blocks,B,corrected.pval,pval_array,estimation=estimation){
  
  source("~/Dropbox/projet spectres/FDA-VisiteA.Pini/code_BlockCovarianceTest/utilities.R")
  n = dim(data)[1]
  p = dim(data)[2]
  pval = matrix(0,nrow=p,ncol=p)
  
  
  points.x = which(blocks %in% ix) # coefficients in block index.x
  data.B1 = data[,points.x] # data of the first block
  
  data.B1.array = array(data=data.B1,dim=c(n,dim(data.B1)[2],B))
  data.B1.list = lapply(seq(dim(data.B1.array)[3]), function(x) data.B1.array[ , , x])
  
  for(iy in 1:(ix-1)){ # y coordinate starting point. stops before the diagonal
    
    # data of the second block
    points.y = which(blocks %in% iy)
    data.B2 = data[,points.y]
    
    index.complement = (1:nblocks)[-c(ix,iy)]
    points.complement = which(blocks %in% index.complement)
    data.complement = data[,points.complement]
    
    # permuted data of the first block
    ncol = dim(data.complement)[2]
    if(ncol > 0){
      data.B1.perm.l = lapply(data.B1.list,permute.conditional,n=n,data.complement=data.complement,estimation=estimation)
    }else{
      data.B1.perm.l = lapply(data.B1.list,permute,n=n)
    }
    
    data.B1.perm = simplify2array(data.B1.perm.l)
    T0.tmp = stat.test(data.B1,data.B2,data.orig=data,points.x=points.x,points.y=points.y)
    
    Tperm.tmp = apply(data.B1.perm,3,stat.test,block2=data.B2,data.orig=data,points.x=points.x,points.y=points.y)
    pval.tmp = mean(Tperm.tmp >= T0.tmp)
    
    pval[points.x,points.y] = pval.tmp
    pval[points.y,points.x] = pval.tmp #symmetrization
    
  }
  return(pval)
}


IWT_Block_precision_unadjusted <- function(data,blocks,B=1000,nworkers=1,estimation='SCAD'){  # estimation method is either SCAD or LM
  
  n <- dim(data)[1]
  p = dim(data)[2]
  nblocks = length(levels(factor(blocks)))
  
  
  # tests on rectangles
  pval_array = array(dim=c(2,p,p))
  corrected.pval = matrix(0,nrow=p,ncol=p)
  
  #responsible.test = matrix(nrow=p,ncol=p)
  #ntests.blocks = zeromatrix = matrix(0,nrow=p,ncol=p)
  
  # For parallelization: the loops are independent between each other, so they can be easily parallelized. 
  # The only point to pay attention to is that for computing the adjusted p-value, the maximization is done at the end of each loop.
  # It would also be possible to save all the results of each loop instead (in an array) and do the maximization only once at the end.
  #for(ix in 2:nblocks){ # x coordinate starting point. 
  #if(nworkers>1){
  #  cl <- makeCluster(nworkers)
  #registerDoParallel(cl)
  #clusterExport(cl,list("permute.conditional","permute"))
  #  }else{
  #   registerDoSEQ()
  #}
  cl <- makeCluster(nworkers)
  result <- clusterMap(cl, blocks_loop_unadjusted, 2:nblocks,
                       MoreArgs=list(nblocks=nblocks,data=data,
                                     blocks=blocks,B=B,corrected.pval=corrected.pval,
                                     pval_array=pval_array,estimation=estimation))
  # result = foreach(ix= 2:nblocks) %dopar%{
  #   library(glmnet)
  #   corrected.pval = matrix(0,nrow=p,ncol=p)
  #   for(lx in 0:(nblocks-ix)){ # length on x axis of the rectangle
  #     index.x = ix:(ix+lx) # index first block
  #     points.x = which(blocks %in% index.x) # coefficients in block index.x
  #     data.B1 = data[,points.x] # data of the first block
  #     
  #     data.B1.array = array(data=data.B1,dim=c(n,dim(data.B1)[2],B))
  #     data.B1.list = lapply(seq(dim(data.B1.array)[3]), function(x) data.B1.array[ , , x])
  #     
  #     for(iy in 1:(ix-1)){ # y coordinate starting point. stops before the diagonal
  #       for(ly in 0:(ix-iy-1)){ # length on y axis of the rectangle
  #         # data of the second block
  #         index.y = iy:(iy+ly) # index second block
  #         points.y = which(blocks %in% index.y)
  #         data.B2 = data[,points.y]
  #         
  #         index.complement = (1:nblocks)[-c(index.x,index.y)]
  #         points.complement = which(blocks %in% index.complement)
  #         data.complement = data[,points.complement]
  #         
  #         # permuted data of the first block
  #         ncol = dim(data.complement)[2]
  #         
  #         if(ncol > 0){
  #           data.B1.perm.l = lapply(data.B1.list,permute.conditional,n=n,data.complement=data.complement)
  #         }else{
  #           data.B1.perm.l = lapply(data.B1.list,permute,n=n)
  #         }
  #         
  #         data.B1.perm = simplify2array(data.B1.perm.l)
  #         T0.tmp = stat.test(data.B1,data.B2,data.orig=data,points.x=points.x,points.y=points.y)
  #         
  #         Tperm.tmp = apply(data.B1.perm,3,stat.test,block2=data.B2,data.orig=data,points.x=points.x,points.y=points.y)
  #         pval.tmp = mean(Tperm.tmp >= T0.tmp)
  #         
  #         corrected.pval_temp = matrix(0,nrow=p,ncol=p)
  #         corrected.pval_temp[points.x,points.y] = pval.tmp
  #         corrected.pval_temp[points.y,points.x] = pval.tmp # simmetrization
  #         
  #         pval_array[1,,] = corrected.pval # old adjusted p-value
  #         pval_array[2,,] = corrected.pval_temp # p-value resulting from the test of this block
  #         
  #         corrected.pval = apply(pval_array,c(2,3),max) # maximization for updating the adjusted p-value
  #       }
  #     }
  #   }
  #   return(corrected.pval)
  # }
  # 
  
  stopCluster(cl)
  
  
  
  pval = apply(simplify2array(result),c(1,2),max) 
  # we still compute the max since result is a list containing the matrices resulting from each line separately 
  # (just one element of the list is different from zero for every block)
  return(pval)
}


