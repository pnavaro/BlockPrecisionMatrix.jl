M = 100 # number of MC simulation replications
alpha = 0.05 # level of the test

# Scenario 1: 1 block close to the diagonal ------------------------------------------------------
p.part = c(10,10,10,10,10,10,10,10,10,10) # size of the blocks: constant for now
p = sum(p.part)
nblocks = length(p.part)

n = 60 # sample size
blocks = rep(1:10,p.part)

s = rep(1,nblocks) # variance in each block
Sigma = matrix(0,nrow=p,ncol=p) # covariance matrix

cov.block = 0.5 # covariance in the block where the covariance is non zero

for(i in 1:p){
  for(j in 1:p){
    if(blocks[i]==blocks[j]){
      Sigma[i,j] = s[blocks[i]]*exp(-(i-j)^2/(p))
    }
    if(blocks[i]==5 & blocks[j]==6 | blocks[i]==6 & blocks[j]==5 ){
      Sigma[i,j] = cov.block
    }
  }
}

mu = rep(0,p)
mu[which(blocks==1|blocks==3)] = 2

library(fields)
image.plot(Sigma)

library(mvtnorm)
data = rmvnorm(n,mu,Sigma)
matplot(t(data),type='l') # instance of generated data

pval.array = unadjusted.pval.array = array(dim=c(M,p,p))
for(sim in 1:M){
  data = rmvnorm(n,rep(0,p),Sigma)
  IWT_Block = IWT_Block_covariance(data,blocks)
  pval.array[sim,,] = IWT_Block$corrected.pval
  unadjusted.pval.array[sim,,] = IWT_Block_covariance_unadjusted(data,blocks)
}

p.rejection = apply((pval.array<alpha),c(2,3),mean)
image.plot(p.rejection)

p.rejection.unadjusted = apply((unadjusted.pval.array<alpha),c(2,3),mean)
image.plot(p.rejection.unadjusted)

image.plot(IWT_Block$ntests.blocks)

save.image('scenario1_all_means.RData')

# Scenario 2: 1 block in the middle ------------------------------------------------------
Sigma = matrix(0,nrow=p,ncol=p)
for(i in 1:p){
  for(j in 1:p){
    if(blocks[i]==blocks[j]){
      Sigma[i,j] = s[blocks[i]]*exp(-(i-j)^2/(p))
    }
    if(blocks[i]==3 & blocks[j]==8 | blocks[i]==8 & blocks[j]==3 ){
      Sigma[i,j] = cov.block
    }
  }
}

library(fields)
image.plot(Sigma)

pval.array = unadjusted.pval.array = array(dim=c(M,p,p))
for(sim in 1:M){
  data = rmvnorm(n,rep(0,p),Sigma)
  pval.array[sim,,] = IWT_Block_covariance(data,blocks)$corrected.pval
  unadjusted.pval.array[sim,,] = IWT_Block_covariance_unadjusted(data,blocks)
}

p.rejection = apply((pval.array<alpha),c(2,3),mean)
image.plot(p.rejection)

p.rejection.unadjusted = apply((unadjusted.pval.array<alpha),c(2,3),mean)
image.plot(p.rejection.unadjusted)

save.image('scenario2_all_means.RData')



# Scenario 3: 1 block in the angle ------------------------------------------------------
Sigma = matrix(0,nrow=p,ncol=p)
for(i in 1:p){
  for(j in 1:p){
    if(blocks[i]==blocks[j]){
      Sigma[i,j] = s[blocks[i]]*exp(-(i-j)^2/(p))
    }
    if(blocks[i]==1 & blocks[j]==10 | blocks[i]==10 & blocks[j]==1 ){
      Sigma[i,j] = cov.block
    }
  }
}

library(fields)
image.plot(Sigma)

pval.array = unadjusted.pval.array = array(dim=c(M,p,p))
for(sim in 1:M){
  data = rmvnorm(n,rep(0,p),Sigma)
  pval.array[sim,,] = IWT_Block_covariance(data,blocks)$corrected.pval
  unadjusted.pval.array[sim,,] = IWT_Block_covariance_unadjusted(data,blocks)
}

p.rejection = apply((pval.array<alpha),c(2,3),mean)
image.plot(p.rejection)

p.rejection.unadjusted = apply((unadjusted.pval.array<alpha),c(2,3),mean)
image.plot(p.rejection.unadjusted)

save.image('scenario3_all_means.RData')




# plots -------------------------------------------------------------------

pdf('power_means.pdf',width=11,height=11)
layout(rbind(1:2,3:4))

load('scenario1_all_means.RData')
image.plot(Sigma,main = 'Scenario 1')
image.plot(var(data),main='Sample covariance one data set',zlim=c(-0.52,1.58))
image.plot(p.rejection,main=paste0('Probability of rejection = ',mean(p.rejection[which(blocks==5),which(blocks==6)])))
image.plot(p.rejection.unadjusted,main=paste0('Probability of rejection = ',mean(p.rejection.unadjusted[which(blocks==5),which(blocks==6)])))

load('scenario2_all_means.RData')
image.plot(Sigma,main = 'Scenario 2')
image.plot(var(data),main='Sample covariance one data set',zlim=c(-0.52,1.58))
image.plot(p.rejection,main=paste0('Probability of rejection = ',mean(p.rejection[which(blocks==3),which(blocks==8)])))
image.plot(p.rejection.unadjusted,main=paste0('Probability of rejection = ',mean(p.rejection.unadjusted[which(blocks==3),which(blocks==8)])))

load('scenario3_all_means.RData')
image.plot(Sigma,main = 'Scenario 3')
image.plot(var(data),main='Sample covariance one data set',zlim=c(-0.52,1.58))
image.plot(p.rejection,main=paste0('Probability of rejection = ',mean(p.rejection[which(blocks==1),which(blocks==10)])))
image.plot(p.rejection.unadjusted,main=paste0('Probability of rejection = ',mean(p.rejection.unadjusted[which(blocks==1),which(blocks==10)])))
dev.off()

# hc ----------------------------------------------------------------------
clustering = hclust(dist(t(data)))
plot(clustering)
hc = cutree(clustering,10)
hc



