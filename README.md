# MCMC_toolkit
Toolkit to work with emcee package. I also included a module that peforms an analytical marginalisation over linear parameters in the likelihood. 
Hence, for a Likelihood with a total of $N$ parameters with $n_{\rm lin}$ linear ones, by using this code you can only sample $N - n_{\rm lin}$. 
It can boost the sampling performance, making the chains to converge quicker. Take a look at the jupyter notebook to see how to do that.
