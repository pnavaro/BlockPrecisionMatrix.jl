using  Random
import Base.Iterators: flatten
using  LinearAlgebra

"""
    cov_simu(rng, blocs, indblocs, blocs_on, D)

# fonction pour simuler des matrices de covariances avec blocs

  - blocs    : indice des intervalles separant les blocs (sortie de StructureCov)
  - indblocs : liste des indices des blocs (sortie de StructureCov)
  - blocs_on  : liste des indices des couples de blocs "allumes"
  - D        : vecteur des valeurs propres de la matrice a simuler
"""
function cov_simu(rng, blocs, indblocs, blocs_on, D)
  
  b = length(blocs) - 1
  p = sum(map(length,indblocs))
  P = zeros(Float64, (p, p))
  
  # matrices de rotation pour les blocs "allumes"
  for i in eachindex(blocs_on)
      B1 = indblocs[blocs_on[i][1]]
      B2 = indblocs[blocs_on[i][2]]
      nB = length(B1) + length(B2)
      P[[B1..., B2...], [B1..., B2...]] .= rotation_matrix(rng, nB)
  end
  
  # matrices de rotation pour les autre blocs centraux (non allumes)
  for i in setdiff(1:b, Iterators.flatten(blocs_on))
      P[indblocs[i], indblocs[i]] .= rotation_matrix(rng, length(indblocs[i]))
  end
  
  for i in 1:length(blocs_on)
    P[indblocs[blocs_on[i][1]] , indblocs[blocs_on[i][2]]] .= 0
  end

  P .= transpose(P)
  
  # Matrice de covariance
  covmat = Hermitian(P' * Diagonal(D) * P)
  
  # Matrice de precision
  premat = P' * Diagonal(1 ./ D) * P
  
  return collect(covmat), premat

end
