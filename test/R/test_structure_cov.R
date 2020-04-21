library(mixAK)

# fonction pour definir les indices des blocs
# p : dimension des donnees
# b : nombre de blocs
# blocsOn : liste des indices des couples de blocs "allumes"
StructureCov = function(p, b = 10, blocsOn, seed = p) {

  set.seed(seed)
  
  blocs = sample(3:(p-2), size = (b-1))
  blocs = sort(blocs)
  blocs = c(1, blocs, p)
  indblocs = list()
  indblocs[[1]] = 1:blocs[2]
  for(i in 2:(length(blocs)-1)) {
    indblocs[[i]] = (blocs[i]+1):blocs[i+1]
  }
  
  matBlocs = matrix(0, ncol = p, nrow = p)
  for (l in 1:length(indblocs)){
    matBlocs[indblocs[[l]], indblocs[[l]]] = 1
  }
  
  for(k in 1:length(blocsOn)) {
    for (i in indblocs[[blocsOn[[k]][1]]]) {
      for (j in indblocs[[blocsOn[[k]][2]]]) {
        matBlocs[i, j] = 1
        matBlocs[j, i] = 1
      }
    }
  }
  
  return(list(blocs = blocs, indblocs = indblocs, matBlocs = matBlocs, blocsOn = blocsOn))
}


# fonction pour simuler des matrices de covariances avec blocs
# blocs    : indice des intervalles separant les blocs (sortie de StructureCov)
# indblocs : liste des indices des blocs (sortie de StructureCov)
# blocsOn  : liste des indices des couples de blocs "allumes"
# D        : vecteur des valeurs propres de la matrice a simuler
CovSimu = function(blocs, indblocs, blocsOn, D) {
  
  set.seed(p)
  
  b = length(blocs) - 1
  p = length(unlist(indblocs))
  P = matrix(0, ncol = p, nrow = p)
  
  # matrices de rotation pour les blocs "allumes"
  for (i in 1:length(blocsOn)){
    B1 = indblocs[[blocsOn[[i]][1]]][1]:indblocs[[blocsOn[[i]][1]]][length(indblocs[[blocsOn[[i]][1]]])]
    B2 = indblocs[[blocsOn[[i]][2]]][1]:indblocs[[blocsOn[[i]][2]]][length(indblocs[[blocsOn[[i]][2]]])]
    nB = length(indblocs[[blocsOn[[i]][1]]])+length(indblocs[[blocsOn[[i]][2]]])
    P[c(B1, B2), c(B1, B2)] = rRotationMatrix(1, nB)
  }
  
  # matrices de rotation pour les autre blocs centraux (non allumes)
  for (i in setdiff(1:b, unlist(blocsOn))) {
    P[indblocs[[i]][1]:indblocs[[i]][length(indblocs[[i]])], indblocs[[i]][1]:indblocs[[i]][length(indblocs[[i]])]] = rRotationMatrix(1, length(indblocs[[i]]))
  }
  
  for (i in 1:length(blocsOn)){
    P[indblocs[[blocsOn[[i]][1]]][1]:indblocs[[blocsOn[[i]][1]]][length(indblocs[[blocsOn[[i]][1]]])] , indblocs[[blocsOn[[i]][2]]][1]:indblocs[[blocsOn[[i]][2]]][length(indblocs[[blocsOn[[i]][2]]])]] = 0
  }
  P = t(P)
  
  # Matrice de covariance
  CovMat = t(P)%*%diag(D)%*%P
  
  # Matrice de precision
  PreMat = t(P)%*%diag(1/D)%*%P
  
  return(list(CovMat = CovMat, PreMat = PreMat))
}

p        = 20 
n        = 500 
b        = 3 
blocsOn  = list(c(1,3))

resBlocs = StructureCov(p, b, blocsOn, seed = 2)
D = runif(p, 10^-4, 10^-2)
resmat = CovSimu(resBlocs$blocs, resBlocs$indblocs, blocsOn, D = D)
