# Marginalize Likelihood

![alt text](https://github.com/tmmergulhao/marginalize-posterior/blob/master/marginalised_vs_nonmarginalised.png?raw=true)

Toolkit to work with emcee package. I also included a module that performs an analytical marginalization over linear parameters in the likelihood. 
Hence, for a Likelihood with a total of $N$ parameters and $n_{\rm lin}$ linear, this code will allow you samples $N - n_{\rm lin}$. 
It can boost the sampling performance, making the chains converge more quicker. The code is supposed to work with any model, so it can be easily modified for your specific likelihood.

Check [these calculations](https://github.com/tmmergulhao/marginalize-posterior/blob/master/Marginalisation%20of%20linear%20parameters.pdf) for the analytical derivations and the attached Jupyter Notebook for some examples.
