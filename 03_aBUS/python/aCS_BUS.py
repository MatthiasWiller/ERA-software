import numpy as np
from ERANataf import ERANataf
from ERADist import ERADist
"""
---------------------------------------------------------------------------
Adaptive conditional sampling algorithm
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Iason Papaioannou (iason.papaioannou@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-01
---------------------------------------------------------------------------
Input:
* N   : number of samples to be generated
* l   : scaling parameter lambda 
* b   : actual intermediate level
* u_j : seeds used to generate the new samples
* H   : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_jk       : next level samples
* geval      : limit state function evaluations of the new samples
* new_lambda : next scaling parameter lambda
* accrate    : acceptance rate of the method
---------------------------------------------------------------------------
Based on:
1."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""
def aCS_BUS(N, lam_prev, h, u_j, log_L_fun, l, gl):
    ## Initialize variables
    #pa = 0.1
    n  = np.size(u_j,axis=0)     # number of uncertain parameters
    Ns = np.size(u_j,axis=1)     # number of seeds
    Na = int(np.ceil(100*Ns/N))  # number of chains after which the proposal is adapted (Na = pa*Ns)

    # number of samples per chain
    Nchain = np.ones((Ns),dtype=int)*int(np.floor(N/Ns))
    Nchain[:np.mod(N,Ns)] = Nchain[:np.mod(N,Ns)]+1

    # initialization
    u_jk   = np.zeros((n,N))   # generated samples 
    leval  = np.zeros((N))   # store lsf evaluations
    acc    = np.zeros((N))   # store acceptance
    mu_acc = np.zeros(int(np.floor(Ns/Na)+1))      # store acceptance
    hat_a  = np.zeros(int(np.floor(Ns/Na)))        # average acceptance rate of the chains
    lam    = np.zeros(int(np.floor(Ns/Na)+1))      # scaling parameter \in (0,1)

    ## 1. compute the standard deviation
    opc = 'b'
    if opc == 'a': # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(n)
    elif opc == 'b': # 1b. sigma = sigma_hat (sample standard deviations)
        mu_hat  = np.mean(u_j,axis=1)    # sample mean
        var_hat = np.zeros(n)     # sample std
        for i in range(n):  # dimensions
            for k in range(Ns):  # samples
                var_hat[i] = var_hat[i] + (u_j[i,k]-mu_hat[i])**2
            var_hat[i] = var_hat[i]/(Ns-1)

        sigma_0 = np.sqrt(var_hat)
    else:
        raise RuntimeError('Choose a or b')

    ## 2. iteration
    star_a = 0.44     # optimal acceptance rate 
    lam[0] = lam_prev # initial scaling parameter \in (0,1)

    # a. compute correlation parameter
    i         = 0                                     # index for adaptation of lambda
    sigma     = np.minimum(lam[i]*sigma_0, np.ones(n))   # Ref. 1 Eq. 23
    rho       = np.sqrt(1-sigma**2)                   # Ref. 1 Eq. 24
    mu_acc[i] = 0 

    # b. apply conditional sampling
    for k in range(Ns):
        idx         = sum(Nchain[:k])           #((k-1)/pa+1)    
        acc[idx]    = 1                         # store acceptance    
        u_jk[:,idx] = u_j[:,k]                  # pick a seed at random
        leval[idx]  = log_L_fun(u_jk[:,idx])    # store the lsf evaluation    
        
        for t in range(1, Nchain[k]):    
            # generate candidate sample        
            v = np.random.normal(loc=rho*u_jk[:,idx+t-1], scale=sigma)
            #v = mvnrnd(rho.*u_jk(:,idx+t-1),diag(sigma.^2))   # n-dimensional Gaussian proposal         
            
            # accept or reject sample
            log_L = log_L_fun(v)          # evaluate loglikelihood function
            ge    = gl(v[-1].reshape(-1), l, log_L)   # evaluate limit state function    
            # He = H(v)
            if ge <= h:
                u_jk[:,idx+t] = v         # accept the candidate in observation region           
                leval[idx+t]  = log_L     # store the loglikelihood evaluation
                acc[idx+t]    = 1         # note the acceptance
            else:
                u_jk[:,idx+t] = u_jk[:,idx+t-1]   # reject the candidate and use the same state
                leval[idx+t]  = leval[idx+t-1]    # store the loglikelihood evaluation    
                acc[idx+t]    = 0                 # note the rejection

        # average of the accepted samples for each seed
        mu_acc[i] = mu_acc[i] + np.minimum(1, np.mean(acc[idx:idx+Nchain[k]])) # min problem
        
        if np.mod(k+1,Na) == 0:
            # c. evaluate average acceptance rate
            hat_a[i] = mu_acc[i]/Na   # Ref. 1 Eq. 25
            
            # d. compute new scaling parameter
            zeta     = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
            lam[i+1] = np.exp(np.log(lam[i]) + zeta*(hat_a[i]-star_a))  # Ref. 1 Eq. 26
            
            # update parameters
            sigma = np.minimum(lam[i+1]*sigma_0, np.ones((n)))  # Ref. 1 Eq. 23
            rho   = np.sqrt(1-sigma**2)                         # Ref. 1 Eq. 24
            
            # update counter
            i = i+1

    # next level lambda
    new_lambda = lam[-1]

    # compute mean acceptance rate of all chains
    accrate = np.mean(hat_a)

    return [u_jk, leval, new_lambda, accrate]
##END
