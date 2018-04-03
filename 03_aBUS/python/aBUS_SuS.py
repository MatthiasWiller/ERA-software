import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from aCS_BUS import aCS_BUS
"""
---------------------------------------------------------------------------
Subset simulation function adapted for aBUS (likelihood input)
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Iason Papaioannou (iason.papaioannou@tum.de)
Implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* N              : number of samples per level
* p0             : conditional probability of each subset
* log_likelihood : log-Likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* cE       : model evidence/marginal likelihood
* c        : scaling constant that holds 1/c >= Lmax
* lam      : scaling parameter lambda (for aCS)
---------------------------------------------------------------------------
Based on:
1."Bayesian inference with subset simulation: strategies and improvements"
   Betz et al. 
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
"""
def aBUS_SuS(N,p0,log_likelihood,distr):
    ## log-likelihood function in standard space
    if isinstance(distr, ERANataf): # use Nataf transform (dependence)
        n = len(distr.Marginals)+1        # number of parameters (dimension)
        log_L_fun = lambda u: log_likelihood(distr.U2X(u[:-1].reshape(-1,1)).flatten())
        # if the samples are standard normal do not make any transform
        if distr.Marginals[0].Name == 'standardnormal':
            log_L_fun = lambda u: log_likelihood(u[:-1])
    
    else: # use distribution information for the transformation (independence)
        n = len(distr.Marginals)+1        # number of parameters (dimension)
        u2x       = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))   # from u to x
        log_L_fun = lambda u: log_likelihood(u2x(u[:-1]))
        
        # if the samples are standard normal do not make any transform
        if distr[0].Name == 'standardnormal':
            log_L_fun = lambda u: log_likelihood(u[:-1])
    
    ## limit state funtion for the observation event (Ref.1 Eq.12)
    gl = lambda pi_u, l, log_L: np.log(sp.stats.norm.cdf(pi_u)) + l - log_L
    # note that gl = log(p) + l(i) - leval
    # where p = normcdf(u_j(end,:)) is the standard uniform variable of BUS

    ## Initialization of variables
    i      = 0                         # number of conditional level
    lam    = 0.6                       # initial scaling parameter \in (0,1)
    max_it = 30                        # maximum number of iterations
    samplesU = {'seeds': list(),
                'total': list()}
    samplesX = list()
    #
    geval = np.zeros(N)                # space for the LSF evaluations
    leval = np.zeros(N)                # space for the LSF evaluations
    nF    = np.zeros(max_it,dtype=int) # space for the number of failure point per level
    h     = np.zeros(max_it)           # space for the intermediate leveles
    prob  = np.zeros(max_it)           # space for the failure probability at each level

    ## aBUS-SuS procedure
    # initial MCS step
    print('Evaluating log-likelihood function ...\t', end='')
    u_j = np.random.normal(size=(n,N))  # N samples from the prior distribution
    for j in range(N):
        leval[j] = log_L_fun(u_j[:,j].reshape(-1,1))  # limit state function in standard (Ref. 2 Eq. 21)

    l = np.max(leval)   # =-log(c) (Ref.1 Alg.5 Part.3)
    print('Done!')

    # SuS stage
    h[i] = np.inf
    while h[i] > 0:
        # increase counter
        i = i+1

        # compute the limit state function (Ref.1 Eq.12)
        geval = gl(u_j[-1,:], l, leval)   # evaluate LSF (Ref.1 Eq.12)

        # sort values in ascending order
        idx        = np.argsort(geval)
        # gsort[j,:] = geval[idx]
        
        # order the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU['total'].append(u_j_sort)   # store the ordered samples

        # intermediate level
        h[i] = np.percentile(geval,p0*100)
        print('\n-Constant c level ', i, ' = ', np.exp(-l))
        print('-Threshold level ', i, ' = ', h[i])

        # number of failure points in the next level
        nF[i] = sum(geval <= max(h[i],0))

        # assign conditional probability to the level
        if h[i] < 0:
            h[i]      = 0
            prob[i-1] = nF[i]/N
        else:
            prob[i-1] = p0      
        
        # select seeds
        samplesU['seeds'].append(u_j_sort[:,:int(nF[i])])       # store ordered level seeds
        
        # randomize the ordering of the samples (to avoid bias)
        idx_rnd   = np.random.permutation(int(nF[i]))
        rnd_seeds = samplesU['seeds'][i-1][:,idx_rnd]     # non-ordered seeds
           
        # sampling process using adaptive conditional sampling
        print('\tMCMC sampling ... \t', end='')
        [u_j,leval,lam,_] = aCS_BUS(N, lam, h[i], rnd_seeds, log_L_fun, l, gl)
        print('Ok!')

        # update the value of the scaling constant (Ref.1 Alg.5 Part.4d)
        l_new = max(l, max(leval))
        h[i]  = h[i] - l + l_new
        l     = l_new
        print('-New constant c level ', i, ' = ', np.exp(-l))
        print('-Modified threshold level ', i, '= ', h[i]) 
        
        # decrease the dependence of the samples (Ref.1 Alg.5 Part.4e)
        p = np.random.uniform(low=np.zeros(N),
                              high=np.min([np.ones(N),
                                           np.exp(leval - l + h[i])],
                                          axis=0))
        u_j[-1,:] = sp.stats.norm.ppf(p) # to the standard space

    # number of intermediate levels
    m = i

    # store final posterior samples
    samplesU['total'].append(u_j)  # store final failure samples (non-ordered)

    # delete unnecesary data
    if m < max_it:
        prob  = prob[:m]
        h     = h[:m]

    ## acceptance probability and evidence (Ref.1 Alg.5 Part.6and7)
    p_acc = np.prod(prob)
    c     = 1/np.exp(l)         # l = -log(c) = 1/max(likelihood)
    cE    = p_acc*np.exp(l)     # exp(l) = max(likelihood)

    ## transform the samples to the physical (original) space
    if isinstance(distr, ERANataf):
        if distr.Marginals[0].Name == 'standardnormal':
            for i in range(m+1):
                p = sp.stats.norm.cdf(samplesU['total'][i][-1,:])
                samplesX.append(np.concatenate((samplesU['total'][i][:-1,:], p.reshape(1,-1)), axis=0))
        else:
            for i in range(m+1):
                p = sp.stats.norm.cdf(samplesU['total'][i][-1,:])
                samplesX.append(np.concatenate((distr.U2X(samplesU['total'][i][:-1,:]), p.reshape(1,-1)), axis=0))
    else:
        if distr[0].Name == 'standardnormal':
            for i in range(m+1):
                p = sp.stats.norm.cdf(samplesU['total'][i][-1,:])
                samplesX.append(np.concatenate((samplesU['total'][i][:-1,:], p.reshape(1,-1)), axis=0))
        else:
            for i in range(m+1):
                p = sp.stats.norm.cdf(samplesU['total'][i][-1,:])
                samplesX.append(np.concatenate((u2x(samplesU['total'][i][:-1,:]), p.reshape(1,-1)), axis=0))
    
    return [h,samplesU,samplesX,cE,c,lam]
##END
