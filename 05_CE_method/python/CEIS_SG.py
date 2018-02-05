import numpy as np
import scipy.stats
from ERANataf import ERANataf
from ERADist import ERADist
from h_calc import h_calc
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
def CEIS_SG(N,rho,f):
    dim = 2

    ## Initialization of variables and storage
    j      = 0                # initial level
    max_it = 100              # estimated number of iterations
    N_tot  = 0                # total number of samples
    
    # Definition of parameters of the random variables (uncorrelated standard normal)
    mu_init = np.zeros(dim)   # ...
    Si_init = np.eye(dim)     # ...
    Pi_init = [1]   # ...
    #
    gamma_hat = np.zeros((max_it+1)) # space for ...

    ## CE Procedure
    # Initializing parameters
    gamma_hat[j] = 1
    mu_hat = mu_init
    Si_hat = Si_init
    Pi_hat = Pi_init

    # Iteration
    for j in range(max_it):
        # Generate samples
        # X = GM_sample(mu_hat, Si_hat, Pi_hat, N)
        X = scipy.stats.multivariate_normal.rvs(mean=mu_hat, cov=Si_hat, size=N)

        # Count generated samples
        N_tot += N

        # Evatluation of the limit state function
        g = f(X.T)

        # Calculating h for the likelihood ratio
        h = h_calc(X, mu_hat, Si_hat, Pi_hat)

        # check convergence
        if gamma_hat[j] == 0:
            k_fin = 1
            break
        
        # obtaining estimator gamma
        gamma_hat[j+1] = np.maximum(0, np.percentile(g, rho*100))

        # Indicator function
        I = (g<=gamma_hat[j+1])

        # Likelihood ratio
        W = scipy.stats.multivariate_normal.pdf(X, mean=np.zeros((dim)), cov=np.eye((dim)))/h

        # Closed-form update
        prod   = np.matmul(W[I], X[I,:])
        summ   = np.sum(W[I])
        mu_hat = (prod)/summ
        Xtmp = X[I,:]-mu_hat
        Xo = (Xtmp)*np.tile(np.sqrt(W[I]),(dim,1)).T
        Si_hat = np.matmul(Xo.T,Xo)/np.sum(W[I]) + 1e-6*np.eye(dim)

    # store the needed steps
    l = j
    
    ## Calculation of the Probability of failure
    W_final = scipy.stats.multivariate_normal.pdf(X, mean=np.zeros(dim), cov=np.eye((dim)))/h
    I_final = (g<=0)
    Pr = 1/N*sum(I_final*W_final)

    return [Pr, l, N_tot, gamma_hat, k_fin]   
##END