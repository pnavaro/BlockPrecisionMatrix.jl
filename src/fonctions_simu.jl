import Base.Iterators: flatten
using  LinearAlgebra

export structure_cov


"""
    structure_cov(p, blocsOn; b = 10, seed = p)

# Fonction pour definir les indices des blocs

  - p : dimension des donnees
  - b : nombre de blocs
  - blocsOn : liste des indices des couples de blocs "allumes"
"""
function structure_cov(rng :: AbstractRNG,
                       p :: Int64, 
                       b :: Int64, 
                       blocsOn :: Vector{Vector{Int64}} )
  
  
  blocs = sample(rng, 3:(p-2), (b-1), replace=false)
  sort!(blocs)
  blocs = [1, blocs..., p]
  indblocs = UnitRange{Int64}[]
  push!(indblocs, 1:blocs[2])

  for i in eachindex(blocs[2:end-1])
    push!(indblocs, (blocs[i]+1):blocs[i+1])
								 end
  
  matblocs = zeros(Int64, (p, p))

  for l in indblocs
      matblocs[l, l] .= 1
  end
  
  for bloc in blocsOn
      for i in indblocs[bloc[1]]
          for j in indblocs[bloc[2]]
              matblocs[i, j] = 1
              matblocs[j, i] = 1
          end
      end
  end
  
  return blocs, indblocs, matblocs

end


"""
    cov_simu(blocs, indblocs, blocs_on, D)

fonction pour simuler des matrices de covariances avec blocs

  - blocs    : indice des intervalles separant les blocs (sortie de [`structure_cov`](@ref))
  - indblocs : liste des indices des blocs (sortie de [`structure_cov`](@ref))
  - blocs_on : liste des indices des couples de blocs "allumes"
  - D        : vecteur des valeurs propres de la matrice a simuler
"""
function cov_simu(blocs, indblocs, blocs_on, D)
  
  b = length(blocs) - 1
  p = sum(map(length,indblocs))
  P = zeros(Float64, (p, p))
  
  # matrices de rotation pour les blocs "allumes"
  for i in eachindex(blocs_on)
      B1 = indblocs[blocs_on[i][1]]
      B2 = indblocs[blocs_on[i][2]]
      nB = length(B1) + length(B2)
      P[[B1..., B2...], [B1..., B2...]] .= RotationMatrix(nB)
  end
  
  # matrices de rotation pour les autre blocs centraux (non allumes)
  for i in setdiff(1:b, flatten(blocs_on))
      P[indblocs[i], indblocs[i]] .= RotationMatrix(length(indblocs[i]))
  end
  
  for bloc in blocs_on
    P[indblocs[bloc[1]] , indblocs[bloc[2]]] .= 0
  end

  P .= transpose(P)
  
  # Matrice de covariance
  covmat = Hermitian(P' * Diagonal(D) * P)
  
  # Matrice de precision
  premat = P' * Diagonal(1 ./ D) * P
  
  return covmat, premat

end
