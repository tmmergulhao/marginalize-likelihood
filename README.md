# Marginalize Posterior

![alt text](https://github.com/tmmergulhao/marginalize-posterior/blob/master/marginalised_vs_nonmarginalised.png?raw=true)

Toolkit to work with emcee package. I also included a module that performs an analytical marginalization over linear parameters in the likelihood. 
Hence, for a Likelihood with a total of $N$ parameters and $n_{\rm lin}$ linear, this code will allow you samples $N - n_{\rm lin}$. 
It can boost the sampling performance, making the chains converge more quicker. Take a look at the jupyter notebook to see how to do that.
