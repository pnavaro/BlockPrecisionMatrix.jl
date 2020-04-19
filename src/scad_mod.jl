"""
    function SCADmod(yvector, x, lambda)

function returning the fitted values pf regression with 
the SCAD penalty (used for conditional permutations)
"""
function scad_mod(yvector, x, lambda)
    
  scad = ncvreg(x, yvector, penalty=:SCAD, lambda=lambda)
    
  n = size(x)[2]
  
  fitted = hcat(ones(n),x) * scad.beta
    
  return fitted

end
