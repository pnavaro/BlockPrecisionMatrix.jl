###########################
# fonctions de simulation #
###########################

using  Random, StatsBase, Rotations
import Base.Iterators: flatten
using  LinearAlgebra

"""
    structure_cov(p, blocsOn; b = 10, seed = p)

# Fonction pour definir les indices des blocs
  - p : dimension des donnees
  - b : nombre de blocs
  - blocsOn : liste des indices des couples de blocs "allumes"
"""
function structure_cov(p, blocsOn; b = 10, seed = p)
  
  rng = MersenneTwister(seed)
  
  blocs = sample(rng, 3:(p-2), size = (b-1), replace=false)
  sort!(blocs)
  blocs = [1, blocs..., p]
  indblocs = Int64[]
  push!(indblocs, 1:blocs[2])
  for i in 2:(length(blocs)-1) 
    push!(indblocs, (blocs[i]+1):blocs[i+1])
  end
  
  matBlocs = zeros(Int64, (p, p))

  for l in 1:length(indblocs)
      matBlocs[indblocs[l], indblocs[l]] .= 1
  end
  
  for k in 1:length(blocsOn)
      for i in indblocs[blocsOn[k][1]]
          for j in indblocs[blocsOn[k][2]]
              matBlocs[i, j] = 1
              matBlocs[j, i] = 1
          end
      end
  end
  
  return Dict(:blocs => blocs, :indblocs => indblocs, 
              :matBlocs => matBlocs, :blocsOn => blocsOn)

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
  for i in 1:length(blocsOn)
    B1 = indblocs[blocsOn[i][1]][1]:indblocs[blocsOn[i][1]][length(indblocs[blocsOn[i][1]])]
    B2 = indblocs[blocsOn[i][2]][1]:indblocs[blocsOn[i][2]][length(indblocs[blocsOn[i][2]])]
    nB = length(indblocs[blocsOn[i][1]])+length(indblocs[blocsOn[i][2]])
    P[[B1..., B2...], [B1..., B2...]] .= rand(RotMatrix{nB})
  end
  
  # matrices de rotation pour les autre blocs centraux (non allumes)
  for i in setdiff(1:b, flatten(blocsOn))
    P[indblocs[[i]][1]:indblocs[[i]][length(indblocs[[i]])], 
      indblocs[[i]][1]:indblocs[[i]][length(indblocs[[i]])]] .= rand(RotMatrix{length(indblocs[i])})
  end
  
  for i in 1:length(blocsOn)
    P[indblocs[blocsOn[i][1]][1]:indblocs[blocsOn[i][1]][length(indblocs[blocsOn[i][1]])] , indblocs[blocsOn[i][2]][1]:indblocs[blocsOn[i][2]][length(indblocs[blocsOn[i][2]])]] .= 0
  end

  transpose!(P)
  
  # Matrice de covariance
  CovMat = P' * Diagonal(D) * P
  
  # Matrice de precision
  PreMat = P' * Diagonal(1/D) * P
  
  return Dict(:CovMat => CovMat, :PreMat => PreMat)

end
