# Marginalize Posterior

![alt text](https://github.com/tmmergulhao/marginalize-posterior/blob/master/marginalised_vs_nonmarginalised.png?raw=true)

Toolkit to work with emcee package. I also included a module that performs an analytical marginalization over linear parameters in the likelihood. 
Hence, for a Likelihood with a total of $N$ parameters and $n_{\rm lin}$ linear, this code will allow you samples $N - n_{\rm lin}$. 
It can boost the sampling performance, making the chains converge more quicker.

Check [these calculations](https://github.com/tmmergulhao/marginalize-posterior/blob/master/Marginalisation%20of%20linear%20parameters.pdf) for the analytical derivations, and check the attached Jupyter Notebook for some examples.
