using  Random, StatsBase
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
function structure_cov(p :: Int64, 
                       b :: Int64, 
                       blocsOn :: Vector{Vector{Int64}}; 
                       seed = 0 :: Int64 )
  
  rng = MersenneTwister(seed)
  
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
  
  for bloc in blocsOn
      for i in indblocs[bloc[1]]
          for j in indblocs[bloc[2]]
              matBlocs[i, j] = 1
              matBlocs[j, i] = 1
          end
      end
  end
  
  return Dict(:blocs => blocs, :indblocs => indblocs, :matBlocs => matBlocs)

end


"""
    cov_simu(blocs, indblocs, blocsOn, D)

# fonction pour simuler des matrices de covariances avec blocs
  - blocs    : indice des intervalles separant les blocs (sortie de StructureCov)
  - indblocs : liste des indices des blocs (sortie de StructureCov)
  - blocsOn  : liste des indices des couples de blocs "allumes"
  - D        : vecteur des valeurs propres de la matrice a simuler
"""
function cov_simu(blocs, indblocs, blocsOn, D)
  
  b = length(blocs) - 1
  p = sum(map(length,indblocs))
  P = zeros(Float64, (p, p))
  
  # matrices de rotation pour les blocs "allumes"
  for i in eachindex(blocsOn)
      B1 = indblocs[blocsOn[i][1]]
      B2 = indblocs[blocsOn[i][2]]
      nB = length(B1) + length(B2)
      P[[B1..., B2...], [B1..., B2...]] .= RotationMatrix(nB)
  end
  
  # matrices de rotation pour les autre blocs centraux (non allumes)
  for i in setdiff(1:b, flatten(blocsOn))
      P[indblocs[i], indblocs[i]] .= RotationMatrix(length(indblocs[i]))
  end
  
  for i in 1:length(blocsOn)
    P[indblocs[blocsOn[i][1]] , indblocs[blocsOn[i][2]]] .= 0
  end

  P .= transpose(P)
  
  # Matrice de covariance
  CovMat = Hermitian(P' * Diagonal(D) * P)
  
  # Matrice de precision
  PreMat = P' * Diagonal(1 ./ D) * P
  
  return Dict(:CovMat => collect(CovMat), :PreMat => PreMat)

end
