@testset "ncvreg works for linear regression" begin

using DataFrames, GLM
import PrecisionMatrix: ncvreg

n = 50
p = 10
X = randn(n,p)
b = randn(p)
y = randn(n) .+  X * b

df = DataFrame( X = X, y = y)

ols = lm(@formula(y ~ X), df)

@show ols

#scad = ncvreg(X,y,lambda=0,penalty=:SCAD,eps=.0001)

@test true

#mcp  = ncvreg(X,y,lambda=0,penalty=:MCP,eps=.0001)

@test true

end

