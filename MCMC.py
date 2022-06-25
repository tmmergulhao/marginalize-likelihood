import json, sys, os, emcee, multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict, Counter
from getdist import plots, MCSamples

#Auxiliary functions
def try_mkdir(name):
	this_dir=os.getcwd()+'/'
	this_dir_to_make = this_dir+name
	#print(this_dir_to_make)
	try:
		os.mkdir(this_dir_to_make)
	except:
		pass

def load_dictionary(filename):
    try:
        with open(filename) as json_file:
            dic = json.load(json_file,object_pairs_hook=OrderedDict)
            
    except:
        print('Problem loading the dictionary with priors!')
        print('Files are supposed to be at', filename)
        print('Check your priors directory, prior filename, and try again.')
        sys.exit(-1)
    return dic

class MCMC():

    def __init__(self,nwalkers,prior_name):
        """Initialize the MCMC class

        Args:
            nwalkers (int): Number of walkers that should be used in the analysis
            prior_name (str): Name of the .json prior name with the parameters boundaries 
        """

        #Initialise the main directories
        self.priors_dir = os.getcwd()+'/priors/'; try_mkdir(self.priors_dir)
        self.chains_dir = os.getcwd()+'/chains/'; try_mkdir(self.chains_dir)
        self.figures_dir = os.getcwd()+"/figures/"; try_mkdir(self.figures_dir)

        #Set an initial number of walkers and the standard burnin fraction
        self.nwalkers = nwalkers
        self.burnin_frac = 0.5

        #Load the prior file and get its specifications
        self.prior_dictionary = load_dictionary(self.priors_dir+prior_name+'.json')
        print('Using ',prior_name,'file')
        self.ndim = len(list(self.prior_dictionary.keys()))
        self.prior_bounds = np.zeros((2,self.ndim))
        self.labels = []
        for x in enumerate(list(self.prior_dictionary.keys())):
            index, key = x
            self.labels.append(key)
            self.prior_bounds[0,index],self.prior_bounds[1,index] = self.prior_dictionary.get(key)

    def ChangeChainsDir(self, new_dir):
        """Change the standard directory to save the chains

        Args:
            new_dir (str): The new directory
        """
        self.chains_dir = new_dir
        print("Chain directory updated. The outputs are going to be saved at:")
        print(self.chains_dir)

    def ChangeFiguresDir(self, new_dir):
        """Change the standard directory to save the figures

        Args:
            new_dir (str): The new directory
        """
        self.figures_dir = new_dir
        print("Figures directory updated. The figures are going to be saved at:")
        print(self.figures_dir)

    def set_walkers(self, nwalkers):
        """Allow the user to change the number of walkers later the class is initialized

        Args:
            nwalkers (int): Number of walkers to be used in the MCMC analysis
        """
        self.nwalkers = nwalkers

    def set_burnin_frac(self, burnin_frac):
        """Allow the user to change the usual burnin_frac used in the analysis. The standard value 
        is 0.5 (i.e, 50% of the chain is discarded away)

        Args:
            burnin_frac (float): The percentage of the chain that must be discarded when performing 
            some analysis
        """
        self.nwalkers = burnin_frac

    def create_walkers(self, mode, file = False, x0 =None, sigmas = None, ranges=None):
        """Create the walkers following three different recipes. Each mode will require a different 
        set of input

        Args:
            mode (str): The name of the recipe to be used. Options: 

            1) 'gaussian': Distribute the walkers following a Gaussian distribution with mean x0 
            (array) and variance sigma (array).You need to give as input the x0 and sigmas

            2) 'uniform': Distribute the walkers following a uniform distribution inside the 
            parameter boundaries defined in the prior file You need to give nothing as input

            'uniform_thin': Same as above, but can choose a different range
            You need to give x0 and ranges as input

            file (bool, optional): Whether to save or not the initial positions in .txt files. 
            Defaults to False.
            x0 ([np.array], optional): Used in the 'gaussian' and 'uniform_thin' recipes. 
            Defaults to None.
            sigmas ([np.array], optional): Used in the 'gaussian' recipe. Defaults to None.
            ranges ([np.array], optional): Used in the 'uniform_thin' recipe. Defaults to None.

        Returns:
            [np.array[nwalkers, nparams]]: A 2D array with the initial position of the walkers
        """

        #Array to storage the position of the walkers
        pos = np.zeros((self.nwalkers,self.ndim))

        #Case in which you create the walkers randomly inside the flat prior bounds
        if (mode == 'uniform_prior'):
            print('Using uniform prior')
            print('\n')
            for i in range(0,self.ndim):
                pos[:,i] = np.random.uniform(self.prior_bounds[0,i],self.prior_bounds[1,i],self.nwalkers)
                print('For param', self.labels[i],':')
                print('Minimum:',round(np.min(pos[:,i]),2),'|','Maximum:',round(np.max(pos[:,i]),2))
                print()

        #Case in which you create the walkers by performing a gaussian sampling with variance sigma (input) around x0 (input)
        if (mode == 'gaussian'):

            for i in range(0,self.ndim):
                pos[:,i] = sigmas[i]*np.random.randn(self.nwalkers) + x0[i]
                print('For param', self.labels[i],':')
                print('Minimum:',round(np.min(pos[:,i]),2),'|','Maximum:',round(np.max(pos[:,i]),2))
                print()
                
        if (mode == 'uniform_thin'):
            print('Using the uniform_thin walker positioning')
            print('\n')
            lower = x0 - ranges
            upper = x0 + ranges

            for i in range(0,self.ndim):
                pos[:,i] = np.random.uniform(lower[i],upper[i],self.nwalkers)
                print('For param', self.labels[i],':')
                print('Minimum:',round(np.min(pos[:,i]),2),'|','Maximum:',round(np.max(pos[:,i]),2))
                print()

        if isinstance(file,str):
            try_mkdir('initial_positions')
            filename = os.getcwd()+'/initial_positions/'+file+'_initial_pos.txt'
            np.savetxt(filename,pos)

        return pos	
            
    #by: F.Beutler
    def gelman_rubin_convergence(self,within_chain_var, mean_chain, chain_length):
        ''' Calculate Gelman & Rubin diagnostic
        # 1. Remove the first half of the current chains
        # 2. Calculate the within chain and between chain variances
        # 3. estimate your variance from the within chain and between chain variance
        # 4. Calculate the potential scale reduction parameter '''
        Nchains = within_chain_var.shape[0]
        dim = within_chain_var.shape[1]
        meanall = np.mean(mean_chain, axis=0)
        W = np.mean(within_chain_var, axis=0)
        B = np.arange(dim,dtype=np.float)
        B.fill(0)
        for jj in range(0, Nchains):
            B = B + chain_length*(meanall - mean_chain[jj])**2/(Nchains-1.)
            estvar = (1. - 1./chain_length)*W + B/chain_length
        return np.sqrt(estvar/W)

    def prep_gelman_rubin(self,sampler):
        chain = sampler.get_chain()
        chain_length = chain.shape[0]
        chainsamples = chain[int(chain_length/2):,:, :].reshape((-1, self.ndim))
        within_chain_var = np.var(chainsamples, axis=0)
        mean_chain = np.mean(chainsamples, axis=0)
        return within_chain_var, mean_chain, chain_length
    ###

    def plot_walkers(self,samplers, name):
        '''
        Get a list of samplers and make the 1D plot. The input must be an array [n_samplers]
        '''
        #Get the number of samplers
        N_samples = len(samplers)

        #Start the figure
        fig, axes = plt.subplots(self.ndim, figsize=(16, self.ndim*3), sharex=True)

        #Get the colors for the plots
        color=cm.brg(np.linspace(0,1,N_samples))

        #Iterate over the backends and plot the walkers
        for i in range(0,N_samples):
            chain = samplers[i].get_chain()
            for index,this_param in enumerate(self.labels):
                ax = axes[index]
                ax.plot(chain[:,:,index],alpha=0.5,color=color[i])
                ax.set_ylabel(this_param,size=35)
            del chain

        fig.tight_layout()
        #directory to figures
        plt.savefig(self.figures_dir+name+'_walkers.pdf')
        plt.close('all')

    def plot_1d(self,samplers, name):
        plot_settings = {
        'ignore_rows':0.5,
        'fine_bins':1000,
        'fine_bins_2D':2000, 
        'smooth_scale_1D':0.3,
                        }
        N_samples = len(samplers)
        samples = []
        chain = samplers[0].get_chain()
        for i in range(0,N_samples):
            chain = samplers[i].get_chain(flat=True)
            samples.append(MCSamples(samples = chain,labels=self.labels,names=self.labels,
            settings = plot_settings))
        del chain
        g1 = plots.get_subplot_plotter(width_inch=20)
        g1.settings.legend_fontsize = 20
        g1.settings.axes_fontsize = 20
        g1.settings.axes_labelsize = 20
        g1.settings.title_limit = True
        g1.plots_1d(samples)

        #Save the figure
        figurehandle = self.figures_dir+name
        g1.export(figurehandle+'1D_ALL.png')
        plt.close('all')

    def plot_corner(self, handle, gelman = None, width_inch = 15, ranges = {},
    plot_settings = {'fine_bins':1000,
    'fine_bins_2D':1500, 'smooth_scale_1D':0.3, 'smooth_scale_2D':0.2}):
        
        if gelman is not None: #If the Gelman-Rubin was used, join the N chains
            N_chains = gelman['N']
            for i in range(0,N_chains):
                name = self.chains_dir+handle+'Run_{}.h5'.format(i)
                backend = emcee.backends.HDFBackend(name, read_only = True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0] #Get the size of the array
                burnin = int(self.burnin_frac*chain_size)
                chain = backend.get_chain(flat=True, discard = burnin)

                if i==0:
                    final_chain = chain

                else:
                    final_chain = np.vstack((final_chain, chain))

        else: #Otherwise, just use a single chain to look for the ML parameters
            name = self.chains_dir+handle+'.h5'
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(self.burnin_frac*chain_size)
            final_chain = backend.get_chain(flat=True, discard = burnin)


        samples = MCSamples(samples = final_chain, labels=self.labels, names=self.labels, 
        settings = plot_settings, ranges = ranges)

        g1 = plots.get_subplot_plotter(width_inch = width_inch)
        g1.settings.legend_fontsize = 20
        g1.settings.axes_fontsize = 20
        g1.settings.axes_labelsize = 20
        g1.settings.title_limit = True
        g1.settings.progress = True
        g1.triangle_plot(samples)

        #Save the figure
        figurehandle = self.figures_dir+handle
        g1.export(figurehandle+'_Corner.png')
        plt.close('all')

    def plot_CorrMatrix(self, handle, gelman = None, figsize=(9,9)):
        """Plot the Correlation Matrix using the MCMC chains

        Args:
            handle (str): The handle used to name the MCMC results
            gelman (dictionary, optional): If there are parallel chains, specify giving as input
            the gelman file you used to run them. Only the key "N" is used. Defaults to None.
            save (string, optional): If want to save the figure, give as input the name of the file. 
            Defaults to None.
            figsize (tuple, optional): The size of the figure. Defaults to (9,9).
        """

        if gelman is not None: #If the Gelman-Rubin was used, join the N chains
            N_chains = gelman['N']
            for i in range(0,N_chains):
                name = self.chains_dir+handle+'Run_{}.h5'.format(i)
                backend = emcee.backends.HDFBackend(name, read_only = True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0] #Get the size of the array
                burnin = int(self.burnin_frac*chain_size)
                chain = backend.get_chain(flat=True, discard = burnin)

                if i==0:
                    final_chain = chain

                else:
                    final_chain = np.vstack((final_chain, chain))

        else: #Otherwise, just use a single chain to look for the ML parameters
            name = self.chains_dir+handle+'.h5'
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(self.burnin_frac*chain_size)
            final_chain = backend.get_chain(flat=True, discard = burnin)

        fig,(ax1) = plt.subplots(1,1, figsize = figsize)

        im = ax1.imshow(np.corrcoef(final_chain.T), cmap = plt.get_cmap('RdBu'))
    
        ax1.set_xticks(np.arange(-0.5,self.ndim-1+1,1),    minor=False)
        ax1.set_yticks(np.arange(-0.5,self.ndim-1+1,1),    minor=False)
        ax1.set_xticklabels([],      minor=False)
        ax1.set_yticklabels([],      minor=False)

        #minor axis - labels
        ax1.set_xticks(np.arange(0,self.ndim,1),    minor=True)
        ax1.set_xticklabels(['$'+x+'$' for x in self.labels],      minor=True)
        ax1.set_yticks(np.arange(0,self.ndim,1),      minor=True)
        ax1.set_yticklabels(['$'+x+'$' for x in self.labels],      minor=True)
        ax1.grid(linewidth=10, color='white')
        fig.colorbar(im)


        plt.savefig(self.figures_dir+handle+'_CorrMatrix.png')
        plt.close('all')
        
    def in_prior(self, theta, params = None):
        """Return True if parameters are inside priors and False otherwise. If you need that only 
        part of the parameters should be tested you must give list with the labels associated to 
        them. The labels must be exactly equal as defined in the prior file.

        Args:
            x (array): The set of parameters to be tested
            params (_type_, optional): _description_. Defaults to None.

        Returns:
            boolean: True if parameters are inside, False otherwise.
        """
        if params is None:
            for i,this_param in enumerate(self.prior_bounds.T):
                this_value = theta[i]
                this_lower_bound,this_upper_bound = this_param
                if(not(this_lower_bound<this_value<this_upper_bound)):
                    return False
            return True
        else:
            for this_param in params:
                this_index = self.labels.index(this_param)
                this_lower_bound, this_upper_bound = self.prior_dictionary[this_param]
                if(not(this_lower_bound<theta[this_index]<this_upper_bound)):
                    return False
            return True

    def log_gaussian_prior(self,x,sigma,params,x0,central=False):
        #Compute the log_gausian prior for a set of parameters
        total_contribution = 0
        for index,this_param in enumerate(params):
            this_index = self.labels.index(this_param)
            this_mean  = x0[index]
            this_entry = x[this_index]
            total_contribution += -0.5*(this_mean-this_entry)**2/sigma[index]**2
        return total_contribution

    def run_MCMC(self,name,steps,pos,loglikelihood,new=True,plots = False,args=None, a=2):
        filename = self.chains_dir+name+'.h5'
        backend  = emcee.backends.HDFBackend(filename)

        #array to storage the autocorr time over steps
        try:
            autocorr = np.loadtxt(self.chains_dir+name+'_tau.txt')
            autocorr = autocorr.tolist()
            acceptance = np.loadtxt(self.chains_dir+name+'_acceptance.txt')
            acceptance = acceptance.tolist()
        except:
            pass

            autocorr = []
            acceptance = []
        
        if args is not None:
            sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,args=args,
            backend=backend,moves=[emcee.moves.StretchMove(a=a)],threads=1)
        else:
            sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,backend=backend,
            moves=[emcee.moves.StretchMove(a=a)],threads = 1)
        
        if new:
            #sampler.run_mcmc(pos,steps,progress=True)
            for sample in sampler.sample(pos, iterations=steps, progress=True):
                if sampler.iteration % 25:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                autocorr.append(np.mean(tau))
                acceptance.append(np.mean(sampler.acceptance_fraction))
                corr_array = np.asarray(autocorr)
                print('Mean acceptance fraction:', np.mean(sampler.acceptance_fraction))
                np.savetxt(self.chains_dir+name+'_tau.txt',corr_array)
                np.savetxt(self.chains_dir+name+'_acceptance.txt',acceptance)
                print('Mean autocorre_time:', np.mean(tau))
                
        else:
            last_sample = sampler.get_last_sample()
            for samp in sampler.sample(last_sample,iterations=steps,progress=True):
                if sampler.iteration % 25:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                autocorr.append(np.mean(tau))
                acceptance.append(np.mean(sampler.acceptance_fraction))
                corr_array = np.asarray(autocorr)
                print('Mean acceptance fraction:', np.mean(sampler.acceptance_fraction))
                np.savetxt(self.chains_dir+name+'_tau.txt',corr_array)
                np.savetxt(self.chains_dir+name+'_acceptance.txt',acceptance)
                print('Mean autocorre_time:', np.mean(tau))

        if plots:
            self.plot_walkers([sampler],name)
            self.plot_1d([sampler],name)
            #self.plot_log_prob([sampler],name)

    def get_chain(self,handle, gelman = None):
        """Search in the total sample of walker positions the set set of parameters that gives the 
        best-fit to the data

        Args:
            handle (str): Handle unsed in the MCMC analysis
            gelman (dic, optional): The dicitonary used as input for the Gelman-Rubin convergence 
            criteria. Defaults to None.

        Returns:
            [np.array[1,nparams] ]: An array with the parameter that gives the better fit to the 
            data
        """

        #If the Gelman-Rubin was used, join the N chains and look for the ML parameters
        if gelman is not None: 
            N_chains = gelman['N']
            for i in range(0,N_chains):
                name = self.chains_dir+handle+'Run_{}.h5'.format(i)
                backend = emcee.backends.HDFBackend(name, read_only = True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0] #Get the size of the array
                burnin = int(self.burnin_frac*chain_size)
                chain = backend.get_chain(flat=True, discard = burnin)
                if i==0:
                    final_chain = chain
                else:
                    final_chain = np.vstack((final_chain, chain))

        else: #Otherwise, just use a single chain to look for the ML parameters
            name = self.chains_dir+handle+'.h5'
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(self.burnin_frac*chain_size)
            final_chain = backend.get_chain(flat=True, discard = burnin)

        return final_chain

    def get_ML(self,handle, gelman = None):
        """Search in the total sample of walker positions the set set of parameters that gives 
        the best-fit to the data

        Args:
            handle (str): Handle unsed in the MCMC analysis
            gelman (dic, optional): The dicitonary used as input for the Gelman-Rubin convergence 
            criteria. Defaults to None.

        Returns:
            [np.array[1,nparams] ]: An array with the parameter that gives the better fit to 
            the data
        """

        #If the Gelman-Rubin was used, join the N chains and look for the ML parameters
        if gelman is not None: 
            N_chains = gelman['N']
            for i in range(0,N_chains):
                name = self.chains_dir+handle+'Run_{}.h5'.format(i)
                backend = emcee.backends.HDFBackend(name, read_only = True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0] #Get the size of the array
                burnin = int(self.burnin_frac*chain_size)
                chain = backend.get_chain(flat=True, discard = burnin)
                logprob = backend.get_log_prob(flat = True, discard = burnin)
                if i==0:
                    final_chain = chain
                    final_logprob = logprob
                else:
                    final_chain = np.vstack((final_chain, chain))
                    final_logprob = np.hstack((final_logprob, logprob))

        else: #Otherwise, just use a single chain to look for the ML parameters
            name = self.chains_dir+handle+'.h5'
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(self.burnin_frac*chain_size)
            final_chain = backend.get_chain(flat=True, discard = burnin)
            final_logprob = backend.get_log_prob(flat = True, discard = burnin)

        index_min = max(range(len(final_logprob)), key=final_logprob.__getitem__)
        ML_params = final_chain[index_min]
        return ML_params

    def get_logprop(self, handle, gelman = None):
        
        if gelman is not None: 
            N_chains = gelman['N']
            for i in range(0,N_chains):
                name = self.chains_dir+handle+'Run_{}.h5'.format(i)
                backend = emcee.backends.HDFBackend(name, read_only = True)
                chain = backend.get_chain(flat=False)
                chain_size = chain.shape[0] #Get the size of the array
                burnin = int(self.burnin_frac*chain_size)
                logprob = backend.get_log_prob(flat = True, discard = burnin)
                if i==0:
                    final_logprob = logprob
                else:
                    final_logprob = np.hstack((final_logprob, logprob))

        else: #Otherwise, just use a single chain to look for the ML parameters
            name = self.chains_dir+handle+'.h5'
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(self.burnin_frac*chain_size)
            final_logprob = backend.get_log_prob(flat = True, discard = burnin)

        #Get the log_prob array
        return final_logprob

    def run_MCMC_MPI(self,name,steps,pos,loglikelihood,pool,new=True,plots = False,args=None):

        filename = self.chains_dir+name+'.h5'
        backend  = emcee.backends.HDFBackend(filename)
        if args is not None:
            sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,args=args,
            backend=backend,pool=pool)
        else:
            sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,backend=backend,
            pool=pool)
        if new:
            sampler.run_mcmc(pos,steps,progress=True)
        else:
            sampler.run_mcmc(None,steps,progress=True)

        if plots:
            self.plot_walkers([sampler],name)
            self.plot_1d([sampler],name)

    def run_MCMC_gelman(self,gelman_rubins,handle,loglikelihood,x0=None,sigmas=None,new_run = True,
    args=None,ranges=None, a=2, save_epsilon = False):
        '''
        This function start the chains and only stop when convergence criteria is achieved
        '''

        #Read the Convergence Parameters from a dictionary
        try:
            N = gelman_rubins['N']
            epsilon = gelman_rubins['epsilon']
            minlength = gelman_rubins['min_length']
            convergence_steps = gelman_rubins['convergence_steps']
            initial_option = gelman_rubins['initial']
        except:
            print('Problem reading the Gelman-Rubin convergence parameters!')
            print('keys: N, epsilon, min_length, convergence_steps, initial')
            sys.exit(-1)

        #List containing all the samplers
        list_samplers = []

        #storate values used to estimate convergence
        within_chain_var = np.zeros((N, self.ndim))
        mean_chain = np.zeros((N, self.ndim))
        chain_length = 0
        scalereduction = np.arange(self.ndim, dtype=np.float)
        scalereduction.fill(2.)

        #Counting the number of iterations:
        counter = 0

        print('You are considering', minlength, 'as the minimum lenght for the chain')
        print('Convergence test happens every', convergence_steps, 'steps')
        print('Number of walkers:', self.nwalkers)
        print('Number of Parameters:', self.ndim)
        print('Number of parallel chains:', N)

        #ask_to_continue()

        #Create all the samplers and their walkers
        for i in range(0,N):
            #create the backend
            filename = self.chains_dir+handle+'Run_'+str(i)+'.h5'
            backend   = emcee.backends.HDFBackend(filename)
            if args is not None:
                list_samplers.append(emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,
                args=args,backend=backend,moves=[emcee.moves.StretchMove(a=a)]))
            else:
                list_samplers.append(emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,
                backend=backend,moves=[emcee.moves.StretchMove(a=a)]))

        #Kicking off all chains to have the minimum length
        if new_run:
            for i in range(0,N):
                to_print = 'Preparing chain '+str(i)
                print(to_print.center(80, '*'))
                name_initial_pos = handle+'_initial_pos_Run_'+str(i)
                print('Positions for the chain', i)
                pos = self.create_walkers(initial_option,file=name_initial_pos,x0=x0,sigmas=sigmas,
                ranges=ranges)
                print('Go!')
                list_samplers[i].run_mcmc(pos,minlength,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = \
                    self.prep_gelman_rubin(list_samplers[i])

        else:
            for i in range(0,N):
                to_print = 'Preparing chain '+str(i)
                print(to_print.center(80, '*'))
                print('Go!')
                list_samplers[i].run_mcmc(None,minlength,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = \
                    self.prep_gelman_rubin(list_samplers[i])
        
        #At this points all chains have the same length. It is checked if they already converged. 
        #If that is not the case they continue to run

        print('All chains with the minimum length!')
        print('Checking convergence...')
        plotname = handle+'_'+str(counter)
        self.plot_1d(list_samplers,plotname)
        scalereduction = self.gelman_rubin_convergence(within_chain_var,mean_chain,chain_length/2)
        eps = abs(1-scalereduction)

        print('epsilon = ', eps)

        if any(eps > epsilon):
            print('Did not converge! Running more steps...')

        
        #If the minimum length was not enough, more steps are done. As soon as the epsilon achieves crosses the threshold, the analysis is done.
        
        while any(eps > epsilon):
            counter += 1
            print('Running iteration', counter)
            for i in range(0,N):
                list_samplers[i].run_mcmc(None,convergence_steps,progress=True)

                within_chain_var[i],mean_chain[i],chain_length = \
                    self.prep_gelman_rubin(list_samplers[i])

            scalereduction = \
                self.gelman_rubin_convergence(within_chain_var,mean_chain,chain_length/2)
            eps = abs(1-scalereduction)

            print('epsilon = ',eps)

            plotname = handle+'_'+str(counter)
            self.plot_1d(list_samplers,plotname)

        print('Convergence Achieved!')
        print('Plotting walkers position over steps...')
        self.plot_walkers(list_samplers,plotname)
        print('Plotting the correlation matrix...')
        self.plot_CorrMatrix(handle = handle, gelman=gelman_rubins, save = handle+"_CorrMatrix",
        figsize = (15,15))
        print('Making a corner plot...')
        self.plot_corner(handle = handle, gelman = gelman_rubins, save = handle+"_Corner", 
        width_inch=25)
        print('Done!')
        
    def run_MCMC_gelman_mpi(self,gelman_rubins,handle,loglikelihood,pool,x0=None,sigmas=None,
    new_run = True, args=None,ranges=None, a=2):
        '''
        This function start the chains and only stop when convergence criteria is achieved
        '''
        #Read the Convergence Parameters from a dictionary
        try:
            N = gelman_rubins['N']
            epsilon = gelman_rubins['epsilon']
            minlength = gelman_rubins['min_length']
            convergence_steps = gelman_rubins['convergence_steps']
            initial_option = gelman_rubins['initial']
        except:
            print('Problem reading the Gelman-Rubin convergence parameters!')
            print('keys: N, epsilon, min_length, convergence_steps, initial')
            sys.exit(-1)

        #List containing all the samplers
        list_samplers = []

        #storate values used to estimate convergence
        within_chain_var = np.zeros((N, self.ndim))
        mean_chain = np.zeros((N, self.ndim))
        chain_length = 0
        scalereduction = np.arange(self.ndim, dtype=np.float)
        scalereduction.fill(2.)

        #Counting the number of iterations:
        counter = 0

        print('You are considering', minlength, 'as the minimum lenght for the chain')
        print('Convergence test happens every', convergence_steps, 'steps')
        print('Number of walkers:', self.nwalkers)
        print('Number of Parameters:', self.ndim)
        print('Number of parallel chains:', N)

        #Create all the samplers and their walkers
        for i in range(0,N):
            #create the backend
            filename = self.chains_dir+handle+'Run_'+str(i)+'.h5'
            backend   = emcee.backends.HDFBackend(filename)
            if args is not None:
                list_samplers.append(emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,
                args=args,backend=backend,pool = pool, moves=[emcee.moves.StretchMove(a=a)]))
            else:
                list_samplers.append(emcee.EnsembleSampler(self.nwalkers,self.ndim,loglikelihood,
                backend=backend, pool = pool, moves=[emcee.moves.StretchMove(a=a)]))

        #Kicking off all chains to have the minimum length
        if new_run:
            for i in range(0,N):
                to_print = 'Preparing chain '+str(i)
                print(to_print.center(80, '*'))
                name_initial_pos = handle+'_initial_pos_Run_'+str(i)
                print('Positions for the chain', i)
                pos = self.create_walkers(initial_option,file=name_initial_pos,x0=x0,sigmas=sigmas,
                ranges=ranges)
                print('Go!')
                list_samplers[i].run_mcmc(pos,minlength,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = self.prep_gelman_rubin(list_samplers[i])

        else:
            for i in range(0,N):
                to_print = 'Preparing chain '+str(i)
                print(to_print.center(80, '*'))
                print('Go!')
                list_samplers[i].run_mcmc(None,minlength,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = \
                    self.prep_gelman_rubin(list_samplers[i])
        
        #At this points all chains have the same length. It is checked if they already converged. 
        #If that is not the case they continue to run
        
        print('All chains with the minimum length!')
        print('Checking convergence...')
        plotname = handle+'_'+str(counter)+'_'
        self.plot_1d(list_samplers,plotname)
        scalereduction = self.gelman_rubin_convergence(within_chain_var,mean_chain,chain_length/2)
        eps = abs(1-scalereduction)

        print('epsilon = ', eps)

        if any(eps > epsilon):
            print('Did not converge! Running more steps...')

        
        #If the minimum length was not enough, more steps are done. As soon as the epsilon achieves 
        # crosses the threshold, the analysis is done.
        
        while any(eps > epsilon):
            counter += 1
            print('Running iteration', counter)
            for i in range(0,N):
                list_samplers[i].run_mcmc(None,convergence_steps,progress=True)
                within_chain_var[i],mean_chain[i],chain_length = \
                    self.prep_gelman_rubin(list_samplers[i])

            scalereduction = \
                self.gelman_rubin_convergence(within_chain_var,mean_chain,chain_length/2)
            eps = abs(1-scalereduction)

            print('epsilon = ',eps)

            self.plot_1d(list_samplers,handle+'_'+str(counter))
        
        print('Convergence Achieved!')
        print('Plotting walkers position over steps...')
        self.plot_walkers(list_samplers, handle)
        print('Plotting the correlation matrix...')
        self.plot_CorrMatrix(handle = handle, gelman=gelman_rubins)
        print('Making a corner plot...')
        self.plot_corner(handle = handle, gelman = gelman_rubins)
        print('Done!')


def get_ML(handle, chain_dir, gelman = None, burnin_frac = 0.5):
    """Search in the total sample of walker positions the set set of parameters that gives the 
    best-fit to the data

    Args:
        handle (str): Handle unsed in the MCMC analysis
        gelman (dic, optional): The dicitonary used as input for the Gelman-Rubin convergence 
        criteria. Defaults to None.

    Returns:
        [np.array[1,nparams] ]: An array with the parameter that gives the better fit to the data
    """

    #If the Gelman-Rubin was used, join the N chains and look for the ML parameters
    if gelman is not None: 
        N_chains = gelman['N']
        for i in range(0,N_chains):
            name = chain_dir+handle+'Run_{}.h5'.format(i)
            #print("Loading:", chain_dir+handle+'Run_{}.h5'.format(i))
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(burnin_frac*chain_size)
            chain = backend.get_chain(flat=True, discard = burnin)
            logprob = backend.get_log_prob(flat = True, discard = burnin)
            if i==0:
                final_chain = chain
                final_logprob = logprob
            else:
                final_chain = np.vstack((final_chain, chain))
                final_logprob = np.hstack((final_logprob, logprob))

    else: #Otherwise, just use a single chain to look for the ML parameters
        name = chain_dir+handle+'.h5'
        backend = emcee.backends.HDFBackend(name, read_only = True)
        chain = backend.get_chain(flat=False)
        chain_size = chain.shape[0] #Get the size of the array
        burnin = int(burnin_frac*chain_size)
        final_chain = backend.get_chain(flat=True, discard = burnin)
        final_logprob = backend.get_log_prob(flat = True, discard = burnin)

    index_min = max(range(len(final_logprob)), key=final_logprob.__getitem__)
    ML_params = final_chain[index_min]
    return ML_params

def get_std(handle, chain_dir, gelman = None, burnin_frac = 0.5):
    """Compute of the standard deviation of effective chain
    
    Args:
        handle (str): Handle unsed in the MCMC analysis
        gelman (dic, optional): The dicitonary used as input for the Gelman-Rubin convergence 
        criteria. Defaults to None.

    Returns:
        [np.array[1,nparams] ]: An array with the standard deviation of all parameters
    """

    #If the Gelman-Rubin was used, join the N chains and look for the ML parameters
    if gelman is not None: 
        N_chains = gelman['N']
        for i in range(0,N_chains):
            name = chain_dir+handle+'Run_{}.h5'.format(i)
            #print("Loading:", chain_dir+handle+'Run_{}.h5'.format(i))
            backend = emcee.backends.HDFBackend(name, read_only = True)
            chain = backend.get_chain(flat=False)
            chain_size = chain.shape[0] #Get the size of the array
            burnin = int(burnin_frac*chain_size)
            chain = backend.get_chain(flat=True, discard = burnin)
            logprob = backend.get_log_prob(flat = True, discard = burnin)
            if i==0:
                final_chain = chain
                final_logprob = logprob
            else:
                final_chain = np.vstack((final_chain, chain))
                final_logprob = np.hstack((final_logprob, logprob))

    else: #Otherwise, just use a single chain to look for the ML parameters
        name = chain_dir+handle+'.h5'
        backend = emcee.backends.HDFBackend(name, read_only = True)
        chain = backend.get_chain(flat=False)
        chain_size = chain.shape[0] #Get the size of the array
        burnin = int(burnin_frac*chain_size)
        final_chain = backend.get_chain(flat=True, discard = burnin)
        final_logprob = backend.get_log_prob(flat = True, discard = burnin)

    return np.std(final_chain, axis = 0)

def main():
    prior = 'FixedCosmo_MT_model1'    
    nwalkers = 30
    MCMC_test = MCMC(nwalkers,prior)

if __name__ == '__main__':
    main()