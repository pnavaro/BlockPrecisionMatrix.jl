p        = 20 # c(20, 100)
n        = 500 # c(100, 200, 500, 1000)
b        = 3 # c(3, 5, 5, 20)
blocsOn  = list(c(1,3))

# simulation des donnees
resBlocs = StructureCov(p, b, blocsOn, seed = 2)
resBlocs$blocs
D        = runif(p, 10^-4, 10^-2)
resmat   = CovSimu(resBlocs$blocs, resBlocs$indblocs, blocsOn, D = D)

#image.plot(resmat$CovMat)
#image.plot(resmat$PreMat)

