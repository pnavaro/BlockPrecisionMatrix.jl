"""
    stat_test(block1_perm, block2, data_orig, points_x, points_y)

We use the Xia estimator for the precision matrix

```R
stat.test = function(block1.perm,block2,data.orig,points.x,points.y){
   data.orig[,points.x] = block1.perm
   PrecMat = PrecXia(data.orig)
   submat = PrecMat\$TprecStd[points.x,points.y]
   return(sum(submat)^2) # We use the Xia estimator for the precision matrix
}
```

"""
function stat_test(block1_perm, block2, data_orig, points_x, points_y)
    
    data_orig[:, points_x] .= block1_perm
    Tprec, TprecStd = prec_xia(data_orig)
    submat = TprecStd[points_x, points_y]
    return sum(submat).^2
    
end

