"""
    stat_test(block1_perm, block2, data_orig, points_x, points_y)

We use the Xia estimator for the precision matrix
"""
function stat_test(block1_perm, block2, data_orig, points_x, points_y)
    
    data_orig[:, points_x] .= block1_perm
    Tprec, TprecStd = prec_xia(data_orig)
    submat = TprecStd[points_x, points_y]
    return sum(submat).^2
    
end

