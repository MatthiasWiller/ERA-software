import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from h_calc import h_calc
"""
---------------------------------------------------------------------------
Cross entropy-based importance sampling with Single Gaussian distribution
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Comments:
* The CE-method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
* General convergence issues can be observed with linear LSFs.
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
def CEIS_SG(N, rho, g_fun, distr):
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
    j      = 0                # initial level
    max_it = 100              # estimated number of iterations
    N_tot  = 0                # total number of samples
    
    # Definition of parameters of the random variables (uncorrelated standard normal)
    mu_init = np.zeros(dim)   # ...
    Si_init = np.eye(dim)     # ...
    Pi_init = [1]             # ...
    #
    gamma_hat = np.zeros((max_it+1)) # space for ...
    samplesU = list()

    # %% CE Procedure
    # Initializing parameters
    gamma_hat[j] = 1
    mu_hat       = mu_init
    Si_hat       = Si_init
    Pi_hat       = Pi_init

    # Iteration
    for j in range(max_it):
        # Generate samples and save them
        X = sp.stats.multivariate_normal.rvs(mean=mu_hat, cov=Si_hat, size=N).reshape(-1,dim)
        samplesU.append(X.T)

        # Count generated samples
        N_tot += N

        # Evaluation of the limit state function
        geval = g(X.T)

        # Calculating h for the likelihood ratio
        h = h_calc(X, mu_hat, Si_hat, Pi_hat)

        # check convergence
        if gamma_hat[j] == 0:
            break
        
        # obtaining estimator gamma
        gamma_hat[j+1] = np.maximum(0, np.percentile(geval, rho*100))

        # Indicator function
        I = (geval<=gamma_hat[j+1])

        # Likelihood ratio
        W = sp.stats.multivariate_normal.pdf(X, mean=np.zeros((dim)), cov=np.eye((dim)))/h

        # Parameter update: Closed-form update
        prod   = np.matmul(W[I], X[I,:])
        summ   = np.sum(W[I])
        mu_hat = (prod)/summ
        Xtmp = X[I,:]-mu_hat
        Xo = (Xtmp)*np.tile(np.sqrt(W[I]),(dim,1)).T
        Si_hat = np.matmul(Xo.T,Xo)/np.sum(W[I]) + 1e-6*np.eye(dim)

    # needed steps
    l = j
    
    # %% Calculation of the Probability of failure
    W_final = sp.stats.multivariate_normal.pdf(X, mean=np.zeros(dim), cov=np.eye((dim)))/h
    I_final = (geval<=0)
    Pr      = 1/N*sum(I_final*W_final)

    k_fin = 1

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
        if distr[0].Name.lower() == 'standardnormal':
            for i in range(l):
                samplesX.append( samplesU[i][:,:] )
        else:
            for i in range(l):
                samplesX.append( u2x(samplesU[i][:,:]) )

    return [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin]
# %%END