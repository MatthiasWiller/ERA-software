import numpy as np
import scipy.stats
from ERANataf import ERANataf
from ERADist import ERADist
from aCS_BUS import aCS_BUS
"""
---------------------------------------------------------------------------
Subset simulation function adapted for BUS (likelihood input)
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
* c        : 1/max(likelihood)
* lam      : scaling of the aCS method
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
def aBUS_SuS(N,p0,log_likelihood,T_nataf):
    if (len(locals()) != 4): # must be the first statement in function
        raise RuntimeError('Incorrect number of parameters'
                           'in function BUS_SuS')

    ## add p Uniform variable of BUS
    n = len(T_nataf.Marginals)+1        # number of parameters (dimension)
    #dist_p  = ERADist('uniform','PAR',[0,1])     # uniform variable in BUS

    ## limit state function in the standard space
    H = lambda u: u[-1] - scipy.stats.invnorm(c*likelihood(T_nataf.U2X(u[1:-2])))
    
    ## Likelihood Function in standard normal space
    log_L_fun = lambda u: log_likelihood(T_nataf.U2X(u[:-1]))
    def lsf(u, log_c):
        geval = np.log(scipy.stats.norm.cdf([u[-1, :]])) - loc_c - log_L(u)
    #lsf = lambda u, log_c: np.log(stats.norm.cdf([u[-1, :]])) - log_c - log_L(u)

    ## Initialization of variables
    i      = 0                         # number of conditional level
    lam    = 0.6                       # initial scaling parameter \in (0,1)
    max_it = 20                        # maximum number of iterations
    samplesU = {'seeds': list(),
                'total': list()}
    samplesX = {'total': list()}
    #
    geval = np.zeros((N))              # space for the LSF evaluations
    leval = np.zeros((N))              # space for the LSF evaluations
    nF    = np.zeros((max_it,1))       # space for the number of failure point per level
    h     = np.zeros((max_it,1))       # space for the intermediate leveles
    prob  = np.zeros((max_it,1))       # space for the failure probability at each level

    ## SuS procedure
    # initial MCS step
    print('Evaluating performance function:\t', end='')
    u_j = np.random.normal(size=(n,N))     # samples in the standard space
    for j in range(N):
        leval[j] = log_L_fun(u_j[:,j])  # limit state function in standard (Ref. 2 Eq. 21)
        if leval[j] <= 0:
            Nf[j] = Nf[j]+1
    l = np.max(leval)           # =-log(c) (Ref.1 Alg.5 Part.3)
    print('OK!')

    # SuS stage
    h[i] = np.inf
    while h[i] > 0:
        # increase counter
        i = i+1

        # compute the limit state function (Ref.1 Eq.12)
        geval = gl(u_j[-1,:], l, leval)   # evaluate LSF (Ref.1 Eq.12)

        # sort values in ascending order
        g_prime = np.sort(geval)
        gsort[j,:] = g_prime
        idx = sorted(range(len(geval)), key=lambda x: geval[x])
        
        # order the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU['total'].append(u_j_sort)   # store the ordered samples

        # intermediate level
        h[i] = np.percentile(gsort[j,:],p0*100)
        fprintf('\n\n-Constant c level ', i, ' = ', np.exp(-l))
        fprintf('\n-Threshold level ', i, ' = ', h[i])

        # number of failure points in the next level
        nF[i] = sum(geval <= max(h[i],0))

        # assign conditional probability to the level
        if h[i] < 0:
            b[i]    = 0
            prob[i] = nF[i]/N
        else:
            prob[i] = p0      
        
        # select seeds
        samplesU['seeds'].append(u_j_sort[:,:nF])       # store ordered level seeds
        
        # randomize the ordering of the samples (to avoid bias)
        idx_rnd   = np.random.permutation(nF)
        rnd_seeds = samplesU['seeds'][i][:,idx_rnd]     # non-ordered seeds
           
        # sampling process using adaptive conditional sampling
        print('MCMC sampling ... \t', end='')
        [u_j,leval,lam,] = aCS_BUS(N, lam, h[i], rnd_seeds, log_L_fun, l, gl)
        print('Ok!')

        # update the value of the scaling constant (Ref.1 Alg.5 Part.4d)
        l_new = max(l, max(leval))
        h[i]  = h[i] - l + l_new
        l     = l_new
        print('\n-New constant c level ', i, ' = ', np.exp(-l))
        print('-Modified threshold level ', i, '= ', h[i]) 
        
        # decrease the dependence of the samples
        P = np.random.uniform(low=np.zeros(N),
                              high=np.min([np.ones(N),
                                           np.exp(leval - ll[i] + h[i])],
                                          axis=0))

        #uj[-1, :] = stats.norm.ppf(p) # the problem is here!!!! 

    m = i
    samplesU['total'].append(u_j)  # store final failure samples (non-ordered)

    # delete unnecesary data
    if m < max_it:
        prob  = prob[:m]
        h     = h[:m]

    ## acceptance probability and evidence
    p_acc = np.prod(prob)
    c     = 1/np.exp(l)         # l = log(1/c) = 1/max(likelihood)
    cE    = p_acc*np.exp(l)     # exp(l) = max(likelihood)

    ## transform the samples to the physical (original) space
    for i in range(m+1):
        #p = dist_p.icdf(scipy.stats.normal.cdf(samplesU['total'][i][-1,:])) is the same as:
        p = scipy.stats.normal.cdf(samplesU['total'][i][-1,:])
        samplesX['total'][i] = [T_nataf.U2X(samplesU['total'][i][:-2,:]), p]
    
    return h,samplesU,samplesX,cE,c,lam
##END
