
#When thinking about measuring deviation, two simple rules are either mean absolute deviation or mean of squared deviations. Today, we largely rely upon the mean of squared deviations.  

#A primary argument for this is based on a paper by Ronald Fisher (Title: A mathematical examination of the methods of determining the accuracy of an observation by the mean error, and by the mean square error)[https://academic.oup.com/mnras/article/80/8/758/1063878] in response to a statement by Eddington.  In short. Fisher demonstrates that when we are estimated population variance of a distribution from a sample variance, that knowing something about mean deviation will provide no additional information about the population variance. Going further however, he also shows that knowing anything about any other metric will not provide any additional information. This property is referred to as sufficiency.  Put differently, when estimating population variance, sample variance is the best metric to use. 
#There are several confusions in relation to this topic so we'll sort those out along the way.  First, let us stipulate that on its face, Fisher's argument is somewhat arbitrary. By that, I mean to say that he has compared the relative value of mean squared deviation with mean absolute deviation, in terms of mean squared deviation, and when estimating population mean squared deviation.  That should seem like a necessarily unfair 'apples to oranges' comparison. And it both is and isn't.  In a vacuum it is. As I'll show below, when mean absolute deviation is used to estimate population mean absolute deviation, it does so better than when mean squared deviation is used to estimate population mean squared deviation. However, Fisher didn't merely say 'in all cases, always, everywhere'. His argument specifically assumed that we're trying to estimate the population mean squared deviation of a normal distribution. That's the key here. So, if we look at the formula for a normal distribution, things start to make sense right away.

#The probability density function (PDF) of the Gaussian distribution, also known as the normal distribution, is given by:

#f(x) = (1 / (σ * sqrt(2π))) * exp(-((x - μ)^2) / (2σ^2))
#where μ is the mean of the distribution and σ is the standard deviation.

#In particular, the formula already includes a term which is equivalent to mean squared deviation. The symbol: σ . However, his argument isn't necessarily true here either as I'll show. In particular, mean absolute deviation is related to mean squared deviation which can be expressed by a constant.  Since sample estimates of population mean absolute deviation form a distribution, all we need to do if figure out what that constant is and add it to our mean absolute deviation to arrive a better estimate of population mean squared deviation than afforded by sample based mean squared deviation.  

##So, the confusion that arises is due to differing statements. Fisher was really saying 'mean squared error is the best method for prediction population standard deviation, therefore mean absolute error or any other rule will be worse.' But, when people think about mean absolute error we're instead focusing on the more general statement about how we should measure deviation conceptually.   



population <- rnorm(10000000,mean=0,sd=10)
population <- population-mean(population)
population <- population/sd(population)

pop.sd <- 1
pop.md <- sum(abs(population-mean(population)))/length(population)
cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,100)
  sds <- sd(samp)
  mds <- sum(abs(samp-mean(samp)))/length(samp)
  return(c(sds,mds))
})
parallel::stopCluster(cl=cl)
res <- t(res)
apply(res,2,var) #sds,mds #conclusion: mean absolute deviation estiamted better. 
#We can confirm visually:
d1 <- density(pop.sd-res[,1])
d2 <- density(pop.md-res[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')
#mean absolute deviation is in red.  We can see that it forms a more narrow distribution with a higher peak.  
#Cool. but what if we want to estimate population standard deviation?
#We can add the constant to mean absolute deviation. 
the.constant <- (1-sqrt(2/pi))
d1 <- density(res[,1]) #standard deviation estimates.
d2 <- density(the.constant+res[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')
#Okay, mean absolute deviation is a better estimator of population standard deviation when we have the constant added. 
#BUT! How do we get the constant? 
#In this case, we used 1- sqrt(2/pi).  the 'sqrt(2/pi)' part is the amount you shrink population standard deviation by to get to population mean absolute error. and the '1' is population standard deviation in this case. So, we're using the thing we're trying to estimate as a basis for constructing the constant.  In the real world, that won't work because we don't know the population values.  We do know the 'sqrt(2/pi)' part and we can know that (1-sqrt(2/pi)) is valid.  But we need to adjust for scale. 
#The fundamental issue here, then, is that we need a method to pick k where k is the value in k*(1-sqrt(2/pi)). It follows from Fisher's argument that it doesn't matter what we try to do because we can't estimate k. 
#I can't think of any obvious, principled, way to do so.  And there may not be one based on Fisher's argument.  However, we might try to construct a prediction rule anyway. Perhaps his argument only applies to rules delimited by some special property that we're under no obligation to confine ourselves to. In fact, if we consider that standard deviations are just a fancy type of mean, then  Stein's estimator tells us that it's possible to get a lower variance estimate when simultaneously estimating 3 parameters than when estimating just 1. ..although doing so may incur a bias. More on that later. For now... 

# if we just take k to be the ratio of sample mean absolute deviation divided by sample mean absolute deviation computed from scaled samples (i.e., standardized), our estimate of population standard deviation from mean absolute deviation is not 14% less efficient.  It's 7-9% less efficient.  So, we're getting closer. let's demonstrate next. 

#To convert directly from mean absolute deviation to standard deviation we we multiply every estimate by sqrt(pi/2). Doing so will allow us to identify the 14% inefficiency pointed out by Fisher. 
cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res2 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,200)
  sds <- sd(samp)
  mds <- sqrt(pi/2)*sum(abs(samp-mean(samp)))/length(samp)
  return(c(sds,mds))
})
parallel::stopCluster(cl=cl)
res2 <- t(res2)

apply(res2,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res2[,2])/var(res2[,1]),'%'))
#I got 15.8% on one run, in the limit of infinitely large samples it's 14%.

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res3 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,100)
  sds <- sd(samp)
  md <- sum(abs(samp-mean(samp)))/length(samp)
  samp.scaled <- scale(samp)
 
  md.scaled <- sum(abs(samp.scaled-mean(samp.scaled)))/length(samp.scaled)
  mds <- (1-sqrt(2/pi))*(md/md.scaled)+sum(abs(samp-mean(samp)))/(length(samp))
  return(c(sds,mds))
})
parallel::stopCluster(cl=cl)
res3 <- t(res3)

apply(res3,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res3[,2])/var(res3[,1]),'%'))


d1 <- density(res3[,1])
d2 <- density(res3[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')

#I got 7.5% on one run, I don't know the limit here but the distribution is consistently lower than 14%.

#Note: the ratio is actually NOT itself constant. It's estimated with each sample and will vary.
#So, the question is whether Fisher's finding is completely true. We can see that 'in the limit' there is a solution that is better than standard deviation. However, it doesn't mean we can necessarily get there without additional information about the population. Furthermore, this approach already relies upon standard deviation to scale the data. Nevertheless, something we can do is go back to our initial example with knowledge of 'the.constant'.  We can see that with a perfect estimate of 'the.constant', we would achieve a 27.2% relative gain in efficiency by using mean absolute deviation.  That's huge. Where we are with 'no knowledge' of the constant is -14%.

print(paste('relative efficiency:',1-var(res[,2]+the.constant)/var(res[,1]),'%'))

#What else:
#- I know that for 1 population, normally distributed, we only need 1 constant.  Estimating that constant would include a variance.... 
#- So, what's most important seems to be the need to estimate the k.

#  One idea would be to simulate several populations with a wide range of variances and then using our md/md.scaled heuristic as the rule, and essentially trying to figure out if we can predict k. 

#For now, we do know that md.scaled should be sqrt(2/pi).  It doesn't end up being that though.  Not sure why exactly, but it seems that random sampling variability results in sometimes overestimating and sometimes in underestimating.  This makes sense because when we multiply our distribution of mean absolute deviations by sqrt(2/pi) ~ 1.253, its variance grows. So then, it seems like what we want to do is only *sometimes* multiply our 1.253 with the a value greater than 1 and at other times, we want a smaller adjustment.   But also, if we use sqrt(2/pi) instead of md.scaled, our efficiency actually gets worse. It essentially reverts back to the -14%. see here: 

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res3 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,100)
  sds <- sd(samp)
  md <- sum(abs(samp-mean(samp)))/length(samp)
  samp.scaled <- scale(samp)
  
  md.scaled <- sqrt(2/pi)#sum(abs(samp.scaled-mean(samp.scaled)))/length(samp.scaled)
  mds <- (1-sqrt(2/pi))*(md/md.scaled)+sum(abs(samp-mean(samp)))/(length(samp))
  return(c(sds,mds))
})
parallel::stopCluster(cl=cl)
res3 <- t(res3)

apply(res3,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res3[,2])/var(res3[,1]),'%'))

d1 <- density(res3[,1])
d2 <- density(res3[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')

#SO, we know what the md.scaled 'should' be in principle and we know that setting it to that value results in worse performance...again, this is tied to the fact that to produce md.scaled, we had to first use standard deviation.... SO then, what if we just exaggerate our distance from that value.  if md.scaled is lower than expected, let's make it multiple times lower, and if it's higher, let's make it multiple times higher.  

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res3 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,100)
  sds <- sd(samp)
  md <- sum(abs(samp-mean(samp)))/length(samp)
  samp.scaled <- scale(samp)
  
  md.scaled <- sum(abs(samp.scaled-mean(samp.scaled)))/length(samp.scaled)
  
  md.scale.off <- 2*(md.scaled-sqrt(2/pi))

  md.scaled <- md.scaled+md.scale.off
  
  mds <- (1-sqrt(2/pi))*(md/md.scaled)+sum(abs(samp-mean(samp)))/(length(samp))
  return(c(sds,mds,md,md.scaled,md.scale.off))
})
parallel::stopCluster(cl=cl)
res3 <- t(res3)

apply(res3,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res3[,2])/var(res3[,1]),'%'))

d1 <- density(res3[,1])
d2 <- density(res3[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')

#Okay, I got -1.8% when I doubled the amount by which md.scaled was either over or under sqrt(2/pi).  SO, perhaps there's something to this.

#Let's also look at a correlation matrix.  Since we're experimenting here it's a good idea to keep more metrics that we can explore.  perahps some of those metrics will provide useful insights about what explains what's going on or help us make other adjustments. 
res4 <- as.data.frame(res3)
res4$error <- 1-res4[,2] #adding error just to aide interpretation
colnames(res4) <- c('sd','md.used','md.baseline','md.scaled','md.scale.off','error')
cor(res4)
#So, when md.scaled and md.scaled.off were smaller, the error was bigger. #Since we want to make the error smaller, perhaps we should just multiply by a bigger number.  This we did 2x before. let's try 3.  Also, that's a linear transformation. Let's look at a plot of the data to see if linear makes the most sense. 
plot(res4$md.scale.off,res4$error) 
lm1 <- lm(res4$error~res4$md.scale.off)
car::qqPlot(lm1)
#It looks like a linear trend might be sufficient here. 

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res3 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,100)
  sds <- sd(samp)
  md <- sum(abs(samp-mean(samp)))/length(samp)
  samp.scaled <- scale(samp)
  
  md.scaled <- sum(abs(samp.scaled-mean(samp.scaled)))/length(samp.scaled)
  
  md.scale.off <- 3*(md.scaled-sqrt(2/pi))
  
  md.scaled <- md.scaled+md.scale.off
  
  mds <- (1-sqrt(2/pi))*(md/md.scaled)+sum(abs(samp-mean(samp)))/(length(samp))
  return(c(sds,mds,md,md.scaled,md.scale.off))
})
parallel::stopCluster(cl=cl)
res3 <- t(res3)

apply(res3,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res3[,2])/var(res3[,1]),'%'))

d1 <- density(res3[,1])
d2 <- density(res3[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')

#Down to under -1% now.... let's increase from 3 to 5. 

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res3 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,100)
  sds <- sd(samp)
  md <- sum(abs(samp-mean(samp)))/length(samp)
  samp.scaled <- scale(samp)
  
  md.scaled <- sum(abs(samp.scaled-mean(samp.scaled)))/length(samp.scaled)
  
  md.scale.off <- 5*(md.scaled-sqrt(2/pi))
  
  md.scaled <- md.scaled+md.scale.off
  
  mds <- (1-sqrt(2/pi))*(md/md.scaled)+sum(abs(samp-mean(samp)))/(length(samp))
  return(c(sds,mds,md,md.scaled,md.scale.off))
})
parallel::stopCluster(cl=cl)
res3 <- t(res3)

apply(res3,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res3[,2])/var(res3[,1]),'%'))

d1 <- density(res3[,1])
d2 <- density(res3[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')

#Back up to -3.85% now.... so there's a limit at which increasing linearly stop helping. when I look at the curve, I can see that the curve is shifted slightly to the right. At 5, we're overestimating. Let's look at diagnostics. 
res4 <- as.data.frame(res3)
res4$error <- 1-res4[,2] #again: adding error just to aide interpretation
colnames(res4) <- c('sd','md.used','md.baseline','md.scaled','md.scale.off','error')
cor(res4)
#Diagnostics confirm. When md.scale.off is larger, error is larger. 
#So let's use a simple rule we'll call 'split the difference'. We go back to 4. If that's better than 3, we go to 4.5. If 4.5 is worse than 4, we back to 4.25. and so on and so forth. 

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res3 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,10000)
  sds <- sd(samp)
  md <- sum(abs(samp-mean(samp)))/length(samp)
  samp.scaled <- scale(samp)
  
  md.scaled <- sum(abs(samp.scaled-mean(samp.scaled)))/length(samp.scaled)
  
  md.scale.off <- 3.1875*(md.scaled-sqrt(2/pi))
  
  md.scaled <- md.scaled+md.scale.off
  
  mds <- (1-sqrt(2/pi))*(md/md.scaled)+sum(abs(samp-mean(samp)))/(length(samp))
  return(c(sds,mds,md,md.scaled,md.scale.off))
})
parallel::stopCluster(cl=cl)
res3 <- t(res3)

apply(res3,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res3[,2])/var(res3[,1]),'%'))

d1 <- density(res3[,1])
d2 <- density(res3[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')
# 4 was worse than 3. slightly overestimating. -1.7%
# 3.5 was slighltly worse than 3. slightly overestimating -0.8%
# 3.25 was close to 3 around -0.79
# 3.125 was better than 3 around -0.66
# 3.0625 was close to 3... there's too much variation to tell anymore.  So, we'll increase the sample sizes to 1000.
# 3.0625 was -0.49, better than 3.
# At 1000, 3 was -0.55
# At 10,000, 3.125 was -0.355.
# At 10,000, 3.0625 was -0.48. 
# At 10,000, 3.1875 was -0.25. 
# Stopping at 3.1875.  (by the way, this routine is implicitly a version of gradient descent.)
res4 <- as.data.frame(res3)
res4$error <- 1-res4[,2] #again: adding error just to aide interpretation
colnames(res4) <- c('sd','md.used','md.baseline','md.scaled','md.scale.off','error')
cor(res4)
#Okay, we're now down to around 0.25% inefficiency from 14%. We can see in the diagnostics that the correlation is now quite small. Further, if we increase the sample size, the inefficiency estimates continue to shrink. 
#However, we have not defeated Fisher's argument. We may be really close to the limit but it's not better than standard deviation and our adjustment does depend on standard deviation. So, it would make sense that our transformation is really just transforming our distribution of mean absolute errors into our distribution of standard deviations, thus negating the efficiency gain by identifying k without standard deviation. Nevertheless, the game here is in figuring out how to apply more or less aggressive adjustments to our estimates in a data driven way rather than simply applying a one-size-fits all rule.   


#But that's not all.  The simulations up to this point have been based on samples where there was no measurement error.  In the real world, it would be foolish to assume that our measurements don't contain any error. So, let's test this out. In the paper,  "Revisiting a 90-year-old debate: the advantages of the mean deviation" there is summary of an argument made by Tukey (1960) (In book: Contributions to probability and statistics; essays in honor of Harold Hotelling. Edited by: Ingram Olkin) and Huber (1981) (pages 2-4 of Title: Robust Statistics).  This argument shows that if just 0.2% of observations are corrupted, then mean absolute deviation will be a better estimator for population standard deviation than standard deviation itself. That is, if just 2 observations in a set of 1,000 observations are corrupted, then the situation is reversed. In specific, they assume that there are two distributions that observations could be drawn from.  One distribution is the uncorrupted distribution  The other is the corrupted distribution.  Both distributions have the same mean.  However, the corrupted distribution has a variance that is 9 times as large as the uncorrupted distribution (which means the standard deviation is 3 times as large [sqrt(9) =3]. In their argument, sqrt(2/pi) is approximated and taken to exactly .8.  In doing so, their is some error in their estimate. But, because we have a computer, we can be more precise  So let's simulate:
population <- rnorm(10000000,mean=0,sd=1)
population2 <- rnorm(10000000,mean=0,sd=3) #change from 3 to other values as desired.

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population','population2'))

res2 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,985)
  samp <- c(samp,sample(population2,20))
  sds <- sd(samp)
  mds <- sqrt(pi/2)*sum(abs(samp-mean(samp)))/length(samp) # sqrt(pi/2) is equal to 1 / sqrt(2/pi). In their papers, 1/.8 becomes 1.25, ours is 1.253314.
  return(c(sds,mds))
})
parallel::stopCluster(cl=cl)
res2 <- t(res2)
apply(res2,2,var) #sd,md.  This shows more variance in md than sd. 
print(paste('relative efficiency:', 1-var(res2[,2])/var(res2[,1]),'%'))

#I got about 1.6% in favor of mean squared error when using 3.  (0.2% corrupted samples)
#But when using 3.1, I got 0.06% in favor of mean absolute error.(0.2% corrupted samples)
#At 3.5, I got 6.5% in favor of mean absolute error. (0.2% corrupted samples)
#At 3.1 with 1% corrupted samples, I got 29.8% in favor of mean absolute error.  
#So, to summarize then, the relative advantage of mean squared error is fickle. It depends on the quality of the data.  HOWEVER, does this example translate well to the real world?  

#There are 2 caveats you won't see in the literature on this topic.  1st: how much corruption are we talking about here?  The example shows 2 observations drawn from a distribution that is 3 times as wide as the original distribution. Is that realistic? Measurement errors do occur but are they often 3x? The best way for us to tell would be to compute the z-scores of each sample and record the most extreme values. This will give us an immediate sense of just how obvious the problem would have to be, relative to the real world because in the real world we can simply compute the z-scores for our sample. So let's do that.  Below I do the same as above with just 2 samples out of 1,000 that come from the corrupted distribution.  I z-score the data, compute the absolute value of the z-scores, then retain the 2 most extreme scores:

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population','population2'))

res2 <- parallel::parSapply(cl=cl,1:10000,function(x){
  #print(x)
  samp <- sample(population,998)
  samp <- c(samp,sample(population2,2))
  #corrupt.id <- sample(1:1000,sample(c(2,0,rep(0,998)),1))# 0.2% of observations corrupted. 
  # if(length(corrupt.id)>0){
  # samp[corrupt.id] <- samp[corrupt.id] + rnorm(length(corrupt.id),0,9*10)} #Corrupted by 3x measurement error. 
 samp <-scale(samp) #converts to z-scores
 samp <- abs(samp)
 zmax <- sort(samp,decreasing = T)[1:2]
  return(zmax)
})
parallel::stopCluster(cl=cl)
summary(c(res2[1,],res2[2,]))
plot(density(res2))
#I got
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#2.613   3.167   3.408   3.731   3.852  10.964 
#A z-score of 3.731, is not uncommon in the context of real-world data.  

#But, it's worse.  We can adjust the parameters above.  If the corrupted distribution is set to just 2x and we take 2% of cases as corrupted, then there's a 3% relative advantage for mean absolute error still and z-scores reduce to about 3.38. Both the 2x adn the 2% are well within the scope of real world scenarios.  

#It is also the case the the argument from Fisher assumed a normal distribution.  Data are not generally going to be perfectly normally distributed.  Let's use the case of a t-distribution.  The t-distribution is very similar to the normal distribution BUT the probability of more extreme values is greater.  Put differently, the tails of the distribution have more substance to them. 
population <- rt(10000000,15)
plot(density(population))
cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterExport(cl=cl,varlist=list('population'))

res2 <- parallel::parSapply(cl=cl,1:10000,function(x){
  samp <- sample(population,100)
  sds <- sd(samp)
  mds <- sqrt(pi/2)*sum(abs(samp-mean(samp)))/(length(samp))
  return(c(sds,mds))
})
parallel::stopCluster(cl=cl)
res2 <- t(res2)

print(paste('relative efficiency:', 1-var(res2[,2])/var(res2[,1]),'%'))
#Okay, I got 6%. BUT, we need to also see *where* the distribution is centered. In the case of the normal distribution, it was very close to the population value. So...
sd(population) #We see the population standard deviation.
apply(res2,2,mean) #sd,md.  In my case, the standard deviation was a closer estimate. 

#Let's look at the distributions as a plot.
d1 <- density(res2[,1])
d2 <- density(res2[,2])
plot(d1,ylim=c(0,max(c(d2$y,d1$y))+1.5),lwd=2,col='blue')
lines(d2,lwd=2,col='red')


#I see that the mean absolute deviation is negatively biased.  It may have lower variance than standard deviation, but it seems to systematically underestimate the population standard deviation.  We could try to eliminate that bias but it might result in more variance just like above when we tried to'shift' the distribution of mean absolute deviation estimates over to the standard deviation location.... this is actually a very simple demonstration of what is known as the 'bias-variance tradeoff.' possibly the simplest you can construct.  Put simply, we want unbiased estimates (i.e., estimates that, on average, reflect the true population parameter) but, the adjustments that we make to our estimator to reduce the bias will typically increase the variance and vice versa.  
#In practice, if our variance is too high, then the uncertainty in our estimates renders them essentially useless. So, we might be willing to accept a small amount of bias IF it substantially reduces our variance.  The opposite can also be true. That is, excessive bias renders our estimate useless because it's a poor reflection of the data generating process (i.e., the population parameter). Worse, if we have multiple parameters that depend on one another, then strong bias will potentially wreak havoc on our inferences. For example, a true correlation between two variable might be positive, but if strong biases are invovled, we could estimate a negative correlation.  



