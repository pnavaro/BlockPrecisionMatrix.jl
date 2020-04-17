using LinearAlgebra
using Random
using PrecisionMatrix
using RCall

rng = MersenneTwister(1234)
   
n, p = 50, 5
# prepare data
X = randn(rng, n, p)              # feature matrix
a0 = collect(1:p)                # ground truths

y = X * a0 + 0.1 * randn(n) # generate response

XX = hcat(X, randn(rng, n, p))

@rput XX

@rput y

R"library(ncvreg)"
R"scad <- coef(ncvreg(XX, y, lambda=0.2, penalty='SCAD', eps=.0001))"

@rget scad

println( " R scad = $scad")

λ = [0.2]

scad = SCAD(XX, y, λ)

println( " Julia scad = $scad")

#=

# solve using linear regression
XX = hcat(ones(n), X)
beta = pinv(XX) * y

# do prediction
yp = XX * beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("LM rmse = $rmse")


@test maximum(abs.(beta .- scad.beta)) < 2e-5

yp = XX * scad.beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("SCAD rmse = $rmse")

mcp = MCP(X, y, λ)

@test maximum(abs.(beta .- mcp.beta)) < 2e-5

yp = XX * mcp.beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("MCP rmse = $rmse")

lasso = Lasso(X, y, λ)

@test maximum(abs.(beta .- lasso.beta)) < 2e-5

yp = XX * lasso.beta 

rmse = sqrt(mean(abs2.(y .- yp)))
println("Lasso rmse = $rmse")

mcp <- coef(ncvreg(X, y, lambda=0, penalty="MCP", eps=.0001))
check(scad, beta, tolerance=.01, check.attributes=FALSE)
check(mcp, beta, tolerance=.01, check.attributes=FALSE)


##########################################
.test = "logLik() is correct: gaussian" ##
##########################################
n <- 50
p <- 10
X <- matrix(rnorm(n*p), ncol=p)
y <- rnorm(n)
fit.mle <- lm(y~X)
fit <- ncvreg(X, y, lambda.min=0)
check(logLik(fit)[100], logLik(fit.mle)[1], check.attributes=FALSE, tol= .001)
check(AIC(fit)[100], AIC(fit.mle), check.attributes=FALSE, tol= .001)

##############################################
.test = "ncvreg reproduces lasso: gaussian" ##
##############################################
require(glmnet)
n <- 50
p <- 10
X <- matrix(rnorm(n*p), ncol=p)

par(mfrow=c(3,2))
y <- rnorm(n)
nlasso <- coef(fit <- ncvreg(X, y, penalty="lasso"))
plot(fit, log=TRUE)
glasso <- as.matrix(coef(fit <- glmnet(X, y, lambda=fit$lambda)))
plot(fit, "lambda")
check(nlasso,  glasso, tolerance=.01, check.attributes=FALSE)

##################################################
.test = "cv.ncvreg() options work for gaussian" ##
##################################################
n <- 50
p <- 10
X <- matrix(rnorm(n*p), ncol=p)
b <- c(-3, 3, rep(0, 8))
y <- rnorm(n, mean=X%*%b, sd=1)

par(mfrow=c(2,2))
cvfit <- cv.ncvreg(X, y)
plot(cvfit, type="all")
print(summary(cvfit))
print(predict(cvfit, type="coefficients"))
print(predict(cvfit, type="vars"))
print(predict(cvfit, type="nvars"))

b <- c(-3, 3, rep(0, 8))
y <- rnorm(n, mean=X%*%b, sd=5)
cvfit <- cv.ncvreg(X, y)
plot(cvfit, type="all")

b <- rep(0, 10)
y <- rnorm(n, mean=X%*%b, sd=5)
cvfit <- cv.ncvreg(X, y)
plot(cvfit, type="all")

###############################################
.test = "ncvreg dependencies work: gaussian" ##
###############################################

# Predict
fit <- ncvreg(X, y, lambda.min=0)
p <- predict(fit, X, 'link', lambda=0.1)
p <- predict(fit, X, 'link')
p <- predict(fit, X, 'response')
p <- predict(fit, X, 'coef')
p <- predict(fit, X, 'vars')
p <- predict(fit, X, 'nvars')

# Integers
X <- matrix(rpois(500, 1), 50, 10)
y <- rpois(50, 1)
fit <- ncvreg(X, y)

# Data frame
fit <- ncvreg(as.data.frame(X), y)

# Penalty factor
fit <- ncvreg(as.data.frame(X), y, penalty.factor=c(0:9))

# User lambdas
fit <- ncvreg(as.data.frame(X), y, lambda=c(1, 0.1, 0.01))

# ReturnX
fit <- ncvreg(as.data.frame(X), y, returnX=TRUE)

# Constant columns
fit <- ncvreg(cbind(5, X), y)

# Plot
plot(fit)
plot(fit, log.l=TRUE)

# Summary
summary(fit, which=10)
summary(fit, lam=0.05)

=#
