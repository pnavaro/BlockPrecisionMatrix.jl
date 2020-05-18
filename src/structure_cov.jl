using Distributions

"""
    structure_cov(rng, p, b, blocs_on)

# Fonction pour definir les indices des blocs

  - p : dimension des donnees
  - b : nombre de blocs
  - blocs_on : liste des indices des couples de blocs "allumes"
"""
function structure_cov(rng :: AbstractRNG,
                       p :: Int, 
                       b :: Int, 
                       blocs_on )
  
  blocs = sample(rng, 3:(p-2), (b-1), replace=false)
  sort!(blocs)
  blocs = [1, blocs..., p]
  indblocs = UnitRange{Int64}[]
  push!(indblocs, 1:blocs[2])

  for i in 2:(length(blocs)-1) 
    push!(indblocs, (blocs[i]+1):blocs[i+1])
  end
  
  matBlocs = zeros(Int64, (p, p))

  for l in indblocs
      matBlocs[l, l] .= 1
  end
  
  for bloc in blocs_on
      for i in indblocs[bloc[1]]
          for j in indblocs[bloc[2]]
              matBlocs[i, j] = 1
              matBlocs[j, i] = 1
          end
      end
  end
  
  return blocs, indblocs, matBlocs

end
