import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from h_calc import h_calc
from GM_sample import GM_sample
from EMGM import EMGM
"""
---------------------------------------------------------------------------
Cross entropy-based importance sampling
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* N     : Number of samples per level
* rho   : cross-correlation coefficient for conditional sampling
* g_fun : limit state function
* distr : Nataf distribution object or
          marginal distribution object of the input variables
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* l         : total number of levels
* N_tot     : total number of samples
* gamma_hat : gamma_ik probability of sample i belonging to distribution k 
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of Gaussians in the mixture
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling 
   using Gaussian densities revisited"
   Geyer et al.
   Engineering Risk Analysis Group, TUM (Sep 2017)
---------------------------------------------------------------------------
"""
def CEIS_GM(N, rho, g_fun, distr):
    # %% initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        dim = len(distr.Marginals)    # number of random variables (dimension)
        g   = lambda u: g_fun(distr.U2X(u))   # LSF in standard space

        # if the samples are standard normal do not make any transform
        if distr.Marginals[0].Name.lower() == 'standardnormal':
            g = g_fun

    elif isinstance(distr, ERADist):   # use distribution information for the transformation (independence)
        dim = len(distr)                    # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))   # from u to x
        g   = lambda u: g_fun(u2x(u))                            # LSF in standard space
        
        # if the samples are standard normal do not make any transform
        if distr[0].Name.lower() =='standardnormal':
            g = g_fun
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')

    # %% Initialization of variables and storage
    j      = 0         # initial level
    max_it = 100       # estimated number of iterations
    N_tot  = 0         # total number of samples
    k      = 1         # number of Gaussians in mixture
    
    # Definition of parameters of the random variables (uncorrelated standard normal)
    mu_init = np.zeros([dim,1])   # ...
    Si_init = np.eye(dim)         # ...
    Pi_init = np.array([1.0])     # ...
    #
    gamma_hat = np.zeros([max_it+1]) # space for ...
    samplesU  = list()

    # %% CE Procedure
    # Initializing parameters
    gamma_hat[j] = 1.0
    mu_hat       = mu_init
    Si_hat       = Si_init
    Pi_hat       = Pi_init

    # Iteration
    for j in range(max_it):
        # Generate samples
        X = GM_sample(mu_hat, Si_hat, Pi_hat, N)
        samplesU.append(X.T)

        # Count generated samples
        N_tot += N

        # Evatluation of the limit state function
        geval = g(X.T)

        # Calculating h for the likelihood ratio
        h = h_calc(X, mu_hat, Si_hat, Pi_hat)

        # check convergence
        if gamma_hat[j] == 0:
            k_fin = k
            break
        
        # obtaining estimator gamma
        gamma_hat[j+1] = np.maximum(0, np.percentile(geval, rho*100))
        print(gamma_hat[j+1])

        # Indicator function
        I = (geval<=gamma_hat[j+1]).astype(int)

        # Likelihood ratio
        W = sp.stats.multivariate_normal.pdf(X, mean=np.zeros((dim)), cov=np.eye((dim)))/h

        # Parameter update: EM algorithm
        nGM = 2
        [mu_hat, Si_hat, Pi_hat] = EMGM(X[I,:].T, W[I], nGM)
        k = len(Pi_hat)

    # store the needed steps
    l = j
    
    # %% Calculation of the Probability of failure
    W_final = sp.stats.multivariate_normal.pdf(X, mean=np.zeros(dim), cov=np.eye((dim)))/h
    I_final = (geval<=0)
    Pr = 1/N*sum(I_final*W_final)
    
    # %% transform the samples to the physical/original space
    samplesX = list()
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        if distr.Marginals[0].Name.lower() == 'standardnormal':
            for i in range(l):
                samplesX.append( samplesU[i][:,:] )
        
        else:
            for i in range(l):
                samplesX.append( distr.U2X(samplesU[i][:,:]) )

    else:
        if distr.Name.lower() == 'standardnormal':
            for i in range(l):
                samplesX.append( samplesU[i][:,:] )
        else:
            for i in range(l):
                samplesX.append( u2x(samplesU[i][:,:]) )

    return [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin]
# %%END