import numpy as np
import scipy.stats
from ERANataf import ERANataf
from ERADist import ERADist
from resampling_index import resampling_index
"""
---------------------------------------------------------------------------
iTMCMC function
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
---------------------------------------------------------------------------
Input:
* Ns             : 
* Nb             : 
* log_likelihood : log-likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* Theta          : 
* q              : 
* S              : 
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements".
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Transitional Markov Chain Monte Carlo method for Bayesian model updating,
   model class selection and model averaging".
   Jianye Ching & Yi-Chun Chen
   Journal of Engineering Mechanics. 133.7 (2007) 816-832
---------------------------------------------------------------------------
"""
def iTMCMC(Ns, Nb, log_likelihood, T_nataf):
    ## some constants and initialization
    d       = len(T_nataf.Marginals)   # number of dimensions
    beta    = 2.4/np.sqrt(d)           # prescribed scaling factor (recommended choice)
    t_acr   = 0.21/d + 0.23            # target acceptance rate
    Na      = 100                      # number of chains to adapt
    thres_p = 1                        # threshold for the c.o.v (100# recommended choice)
    max_it  = 20                       # max number of iterations (for allocation)
    S       = np.ones(max_it)      # space for factors S_j
    q       = np.zeros(max_it)     # store tempering parameters
    j       = 0                        # initialize counter for intermediate levels
    Theta = {'standard': list(),
             'original': list()}


    ## 1. Obtain N samples from the prior pdf and evaluate likelihood
    u_j     = scipy.stats.norm.rvs(size=(Ns,d))          # u_0 (Nxd matrix)
    theta_j = T_nataf.U2X(u_j).T    # theta_0 (transform the samples)
    logL_j  = np.zeros(Ns)
    for i in range(Ns):
        logL_j[i] = log_likelihood(theta_j[i,:])
    
    Theta['standard'].append(u_j)           # store initial level samples
    Theta['original'].append(theta_j)       # store initial level samples

    ## iTMCMC
    while q[j] < 1:   # adaptively choose q
        j = j+1
        print('\niTMCMC intermediate level j = ', j-1,', with q_{j} = ', q[j], '\n')
        
        # 2. Compute tempering parameter p_{j+1}
        # e   = p_{j+1}-p_{j}
        # w_j = likelihood^(e), but we are using the log_likelihood, then:
        fun = lambda e: np.std(np.exp(np.abs(e)*logL_j)) - thres_p*np.mean(np.exp(np.abs(e)*logL_j))   # c.o.v equation
        # [e,~,flag] = fzero(fun, 0)
        e = scipy.optimize.fsolve(fun, 0)

        #
        if e != np.nan:
            q[j+1] = np.minimum(1, q[j]+e)
        else:
            q[j+1] = 1
            print('Variable q was set to ',q[j+1],', since it is not possible to find a suitable value')
    
        # 3. Compute 'plausibility weights' w(theta_j) and factors 'S_j' for the evidence
        w_j  = np.exp((q[j+1]-q[j])*logL_j)    # [Ref. 2 Eq. 12]
        S[j] = np.mean(w_j)                    # [Ref. 2 Eq. 15]
        
        # 4. Metropolis resampling step to obtain N samples from f_{j+1}(theta)
        # weighted sample mean
        w_j_norm = w_j/np.sum(w_j)             # normalized weights
        mu_j     = np.matmul(u_j.T,w_j_norm)   # sum inside [Ref. 2 Eq. 17]
        
        # Compute scaled sample covariance matrix of f_{j+1}(theta)
        Sigma_j = np.zeros(d)
        for k in range(Ns):
            tk_mu   = u_j[k,:] - mu_j
            Sigma_j = Sigma_j + w_j_norm[k]*(np.matmul(tk_mu.T,tk_mu))   # [Ref. 2 Eq. 17]
        
        # target pdf \propto prior(\theta)*likelihood(\theta)^p(j)
        level_post = lambda t, log_L: T_nataf.jointpdf(t)*np.exp(q[j+1]*log_L) # [Ref. 2 Eq. 11]
        
        # Start N different Markov chains
        msg     = '* M-H sampling...\n'
        u_c     = u_j
        theta_c = theta_j
        logL_c  = logL_j
        Nadapt  = 1
        na      = 0
        acpt    = 0
        for k in range(Ns+Nb):      
            # select index l with probability w_j
            l = resampling_index(w_j)
            
            # sampling from the proposal and evaluate likelihood
            u_star     = scipy.stats.norm.rvs(u_c[l,:],(beta**2)*Sigma_j)
            theta_star = T_nataf.U2X(u_star.reshape(-1,1)).T  # transform the sample
            logL_star  = log_likelihood(theta_star.flatten())
            
            # compute the Metropolis ratio
            ratio = level_post(theta_star  ,logL_star)/ \
                    level_post(theta_c[l,:],logL_c[l])
            
            # accept/reject step
            if scipy.stats.uniform.rvs() <= ratio:
                u_c[l,:]     = u_star
                theta_c[l,:] = theta_star
                logL_c[l]    = logL_star
                acpt         = acpt+1
            
            if k > Nb:   # (Ref. 1 Modification 2: burn-in period)
                u_j[k-Nb,:]     = u_c[l,:]
                theta_j[k-Nb,:] = theta_c[l,:]
                logL_j[k-Nb]    = logL_c[l]
            
            # recompute the weights (Ref. 1 Modification 1: update sample weights)
            w_j[l] = np.exp((q[j+1]-q[j])*logL_c[l])
            
            # adapt beta (Ref. 1 Modification 3: adapt beta)
            na = na+1
            if na >= Na:
                p_acr  = acpt/Na
                ca     = (p_acr - t_acr)/np.sqrt(Nadapt)
                beta   = beta*np.exp(ca)
                Nadapt = Nadapt+1
                na     = 0
                acpt   = 0         

        print(np.tile('\b',1,msg)) # TODO: what does this do?
        
        # store samples
        Theta['standard'].append(u_j)
        Theta['original'].append(theta_j)

    # delete unnecesary data
    if j < max_it:
        q = q[:j+2]
        S = S[:j+1]

    ## Compute evidence (normalization constant in Bayes' theorem)
    S = np.prod(S)   # [Ref. 2 Eq. 17]

    return [Theta, q, S]
## END