# PrecisionMatrix.jl

Documentation for PrecisionMatrix.jl

[Explication animée du principe des tests par permutations](https://www.jwilber.me/permutationtest/)

Dans l'animation, ils illustrent un test sur la moyenne. 

Dans notre cas, on veut appliquer les idées des tests par permutation (inverse de la matrice de covariance) mais pour identifier les blocs nuls d'une matrice de précision. Un bloc sera nul dans la situation suivante.

Considérons les blocs associés aux variables X et Y d'un ensemble de données D tel que

``D = (X,Y,Z)`` avec ``X \in R^{n,p_X}, Y \in R^{n,p_Y}, Z \in R^{n,p_Z}``

```math
\begin{aligned}
X & = \beta_X [1,Z] + \epsilon_X \\
Y & = \beta_Y [1,Z] + \epsilon_Y
\end{aligned}
```

alors le bloc ``P_{XY}`` est nul si ``\epsilon_X`` et ``\epsilon_Y`` sont indépendants ie si la matrice de corrélation de ``\epsilon_X`` avec ``\epsilon_Y`` est nulle.

Du coup on applique les  permutations aux résidus ``\epsilon_X`` et/ou
``\epsilon_Y``. Pour faire ça, on a d'abord besoin d'estimer le modèles
linéaire (c'est ici qu'on utilise l'estimation avec SCAD), de
récupérer les résidus, de les permuter puis de reconstruire les
données corresopndant aux résidus permutés (-> fonction
`permute.conditional`)

```R
  permute.conditional = function(y,n,data.complement,estimation){ # uses SCAD/OLS
    permutation = sample(n)
    
    # SCAD/OLS estimaytion
    fitted = switch(estimation, 
                    LM=lm.fit(cbind(1,data.complement),y)$fitted,
                    SCAD=apply(y,2,SCADmod,x=data.complement,lambda=2*sqrt(var(y[,1])*log(ncol(data.complement))/nrow(data.complement))))
    
    residuals = y - fitted
    
    result = fitted + residuals[permutation,]
    return(result) 
  }
```

Pour finir le test, on a besion d'esitmer les matrices de précision associées aux permutations (-> fonction stat.test)

```R
  stat.test = function(block1.perm,block2,data.orig,points.x,points.y){
    data.orig[,points.x] = block1.perm
    PrecMat = PrecXia(data.orig)
    #Rhohat = cor(block1,block2,method='pearson')
    #Rohat.std = Rhohat^2/(1-Rhohat^2)
    submat = PrecMat$TprecStd[points.x,points.y]
    return(sum(submat)^2) # We use the Xia estimator for the precision matrix
  }
```

