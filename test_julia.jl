using Distributions

p        = 20   # (20, 100)
n        = 500  # (100, 200, 500, 1000)
b        = 3    # (3, 5, 5, 20)
blocsOn  = [[1,3]]

# simulation des donnees
resBlocs = structure_cov(p, b, blocsOn, seed = 2)
resBlocs[:blocs]
D        = rand(Uniform(1e-4, 1e-2), p)
resmat   = CovSimu(resBlocs$blocs, resBlocs$indblocs, blocsOn, D = D)
image.plot(resmat$CovMat)
image.plot(resmat$PreMat)

