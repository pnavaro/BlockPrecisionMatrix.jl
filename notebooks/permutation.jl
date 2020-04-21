# -*- coding: utf-8 -*-
# Considérons les blocs associés aux variables X et Y d'un ensemble de données D tel que
#
#
# $D = (X,Y,Z)$ avec $X \in R^{n,p_X}, Y \in R^{n,p_Y}, Z \in R^{n,p_Z} $

# +
using CategoricalArrays
using Random, LinearAlgebra, Distributions
using GLMNet
using InvertedIndices
using NCVREG
using Plots

include("../src/rotation_matrix.jl")
include("../src/fonctions_simu.jl")

p        = 20
n        = 500
b        = 3
blocs_on = [[1,3]]

rng = MersenneTwister(42)

blocs, indblocs, matblocs = structure_cov(rng, p, b, blocs_on)
# -

D        = rand(Uniform(1e-4, 1e-2), p)
covmat, premat   = cov_simu(blocs, indblocs, blocsOn, D)

heatmap(covmat, c=ColorGradient([:red,:blue]))



# $$
# \begin{aligned}
# X = \beta_X [1,Z] + \epsilon_X \\
# Y = \beta_Y [1,Z] + \epsilon_Y
# \end{aligned}
# $$
#
# alors le bloc $P_{XY}$ est nul si $\epsilon_X$ et $\epsilon_Y$ sont indépendants ie si la matrice de corrélation de $\epsilon_X$ avec $\epsilon_Y$ est nulle.
#
# Du coup on applique les  permutations aux résidus $\epsilon_X$ et/ou $\epsilon_Y$. Pour faire ça, on a d'abord besoin d'estimer le modèles linéaire (c'est ici qu'on utilise l'estimation avec SCAD), de récupérer les résidus, de les permuter puis de reconstruire les données correspondant aux résidus permutés (-> fonction permute.conditional)
#
# ```R
#   permute.conditional = function(y,n,data.complement,estimation){ # uses SCAD/OLS
#     permutation = sample(n)
#     
#     # SCAD/OLS estimation
#     fitted = switch(estimation, 
#                     LM=lm.fit(cbind(1,data.complement),y)$fitted,
#                     SCAD=apply(y,2,SCADmod,x=data.complement,lambda=2*sqrt(var(y[,1])*log(ncol(data.complement))/nrow(data.complement))))
#     
#     residuals = y - fitted
#     
#     result = fitted + residuals[permutation,]
#     return(result) 
#   }
# ```

# Pour finir le test, on a besoin d'estimer les matrices de précision associées aux permutations 
#
# ```R
#   stat.test = function(block1.perm,block2,data.orig,points.x,points.y){
#     data.orig[,points.x] = block1.perm
#     PrecMat = PrecXia(data.orig)
#     submat = PrecMat$TprecStd[points.x,points.y]
#     return(sum(submat)^2) # We use the Xia estimator for the precision matrix
#   }
# ```


