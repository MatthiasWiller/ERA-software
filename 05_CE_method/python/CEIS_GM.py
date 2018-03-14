import numpy as np
import scipy.stats
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
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
---------------------------------------------------------------------------
Input:

---------------------------------------------------------------------------
Output:

---------------------------------------------------------------------------
Based on:

---------------------------------------------------------------------------
"""
def CEIS_GM(N, rho, g_fun, distr):
    ## initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        dim = len(distr.Marginals)    # number of random variables (dimension)
        g   = lambda u: g_fun(distr.U2X(u))   # LSF in standard space

        # if the samples are standard normal do not make any transform
        if distr.Marginals[0].Name.lower() == 'standardnormal':
            g = g_fun

    elif isinstance(distr, ERADist):   # use distribution information for the transformation (independence)
        dim = len(distr)                    # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(scipy.stats.norm.cdf(u))   # from u to x
        g   = lambda u: g_fun(u2x(u))                            # LSF in standard space
        
        # if the samples are standard normal do not make any transform
        if distr[0].Name.lower() =='standardnormal':
            g = g_fun
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')

    ## Initialization of variables and storage
    j      = 0         # initial level
    max_it = 100       # estimated number of iterations
    N_tot  = 0         # total number of samples
    k      = 1         # number of Gaussians in mixture
    
    # Definition of parameters of the random variables (uncorrelated standard normal)
    mu_init = np.zeros(dim)   # ...
    Si_init = np.eye(dim)     # ...
    Pi_init = [1]   # ...
    #
    gamma_hat = np.zeros((max_it+1)) # space for ...
    samplesU  = list()

    ## CE Procedure
    # Initializing parameters
    gamma_hat[j] = 1
    mu_hat = mu_init
    Si_hat = Si_init
    Pi_hat = Pi_init

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
        I = (geval<=gamma_hat[j+1])

        # Likelihood ratio
        W = scipy.stats.multivariate_normal.pdf(X, mean=np.zeros((dim)), cov=np.eye((dim)))/h

        # Parameter update: EM algorithm
        nGM = 3

        [mu_hat, Si_hat, Pi_hat, k] = EMGM(X[I,:].T, W[I], nGM)

    # store the needed steps
    l = j
    
    ## Calculation of the Probability of failure
    W_final = scipy.stats.multivariate_normal.pdf(X, mean=np.zeros(dim), cov=np.eye((dim)))/h
    I_final = (geval<=0)
    Pr = 1/N*sum(I_final*W_final)
    
    ## transform the samples to the physical/original space
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
##END