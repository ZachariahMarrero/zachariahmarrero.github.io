#Here's an example R code that computes the density of a vector using the Gaussian kernel with a bandwidth of 2:
# Define the kernel function


#This is just the probability density function for a Gaussian distribution
kernel_function <- function(x,mu=0,sigma=1) {
  (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu)^2 / (2 * sigma^2)))
}

#The next step is to define a density function to compute the densities for each point. This function takes three arguments: `x` is a vector of values at which to compute the density estimate, `data` is the vector of data points, and `bandwidth` is the bandwidth parameter.

# Define the density function using the kernel function and bandwidth
#Bandwidth has the same effect as changing sigma in the Gaussian distribution function.  So sigma = 2 is the same as bandwidth = 2. Setting both to 2 is like saying sigma = 4. However, you could use a different kernel such as a triangular kernel. In that case, bandwidth would have a different effect.

#The `density_function` function loops over each data point, computes the contribution of each point to the estimated density using the kernel function, and then sums up these contributions to produce the final density estimate. The density estimate is then divided by the total number of data points multiplied by the bandwidth to normalize the estimate. So in the end we get back length(x) number of points at which the density was estimated, using each member of 'data.'
density_function <- function(x, data, bandwidth) {
  n <- length(data)
  density <- rep(0, length(x))
  for (i in seq(n)) {
    density <- density + kernel_function((x - data[i]) / bandwidth)
  }
  density <- density / (n * bandwidth)
  return(density)
}

# Generate some data

data <- rnorm(100, mean = 0, sd = 1)

# Compute the density using the function
x <- seq(-10, 10, length.out = 1000) #these are the points at which the kernel is set. internally, the density of each point in 'data' is found by computing the density relative to each of the members in x.  These values are then summed and normalized so that area underneath the curve integrates to 1, which is important if we want to define probability as being between 0 and 1. This does not mean a density cannot have a value greater than 1.  Densities are not probabilities. Although not literally, a density be thought of as a probability multiplied by a range. For example, the proportion of cases in a histogram bar multiplied by the width of the bar. 

#One way to think about a probability density function is to imagine dividing the range of the variable into small intervals, and then calculating the probability that the values of the variable fall within each interval, similar ot a histogram. The width of each interval is like a "bandwidth" or "range" of values, and the probability for each interval is like a "probability density." If you multiply the probability density for each interval by the width of the interval, you get the probability that the variable falls within that interval. Summing up the probabilities for all the intervals gives you the total probability that the variable falls within the entire range.

#Testing the density function: 
bandwidth <- 2
density <- density_function(x, data, bandwidth)

# Plot the results
d1 <- density(data,bw=2) #bw = bandwidth

#Now plot the results to see how the two compare.  
#In blue will be the results from the function constructed here. 
#In red will be the results of the base R, rnorm() function. 

#xlim controls the limits of the plot. 
#lwd controls the width of the plotted line.
#type="l". This is saying to print a line instead of a scatterbplot.
#main. This is the title of the plot. 

plot(x, density,xlim=c(min(d1$x),max(d1$x)), type = "l", main = "Kernel Density Estimate plot",col='blue',lwd=2)

#lines converts scatterplots to a line graph.  You need to make sure the points are ordered correctly as they are joined one point to the next.  In this case that is already done implicitly with how our function was constructed.  If it wasn't, we would sort the values according to their x values, lowest to highest. 
lines(density(data,bw=2),xlim=c(min(d1$x),max(d1$x)),col='red',lwd=2) 

