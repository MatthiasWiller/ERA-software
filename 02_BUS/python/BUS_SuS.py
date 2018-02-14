import numpy as np
import scipy.stats
from ERANataf import ERANataf
from ERADist import ERADist
from aCS import aCS
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
* N         : number of samples per level
* p0        : conditional probability of each subset
* c         : scaling constant of the BUS method
* likelihood: likelihood function of the problem at hand
* T_nataf   : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* b        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* cE       : model evidence/marginal likelihood
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
"""
def BUS_SuS(N,p0,c,likelihood,T_nataf):
    ## add p Uniform variable of BUS
    n = len(T_nataf.Marginals)+1        # number of parameters (dimension)
    #dist_p  = ERADist('uniform','PAR',[0,1])     # uniform variable in BUS

    ## limit state function in the standard space
    # H = lambda u: u[-1] - scipy.stats.norm.ppf(c*likelihood(T_nataf.U2X(u[0:-2])))
    def H(u):
        tmp1 = T_nataf.U2X(u[0:-1].reshape((-1,1))).flatten()
        tmp2 = likelihood(tmp1)
        tmp3 = c*tmp2
        tmp4 = scipy.stats.norm.ppf(tmp3)
        tmp5 = u[-1] - tmp4
        return tmp5

    ## Initialization of variables
    j      = 0                         # number of conditional level
    lam    = 0.6                       # initial scaling parameter \in (0,1)
    max_it = 20                        # maximum number of iterations
    samplesU = {'seeds': list(),
                'total': list()}
    samplesX = {'total': list()}
    #
    geval = np.zeros(N)            # space for the LSF evaluations
    gsort = np.zeros((max_it,N))   # space for the sorted LSF evaluations
    Nf    = np.zeros(max_it)       # space for the number of failure points per level
    b     = np.zeros(max_it)       # space for the intermediate leveles
    prob  = np.zeros(max_it)       # space for the failure probability at each level

    ## SuS procedure
    # initial MCS step
    print('Evaluating performance function:\t', end='')
    u_j = scipy.stats.norm.rvs(size=(n,N))     # samples in the standard space
    for i in range(N):
        geval[i] = H(u_j[:,i])    # limit state function in standard (Ref. 2 Eq. 21)
        if geval[i] <= 0:
            Nf[j] = Nf[j]+1
    print('OK!')

    # SuS stage
    while True:
        # sort values in ascending order
        idx        = np.argsort(geval)
        gsort[j,:] = geval[idx]
        
        # order the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU['total'].append(u_j_sort)   # store the ordered samples

        # intermediate level
        b[j] = np.percentile(gsort[j,:].flatten(),p0*100)
        
        # number of failure points in the next level
        nF = sum(gsort[j,:] <= max(b[j],0))

        # assign conditional probability to the level
        if b[j] <= 0:
            b[j]    = 0
            prob[j] = nF/N
        else:
            prob[j] = p0      
        print('\n-Threshold intermediate level ', j, ' = ', b[j])
        
        # select seeds
        samplesU['seeds'].append(u_j_sort[:,:nF])       # store ordered level seeds
        
        # randomize the ordering of the samples (to avoid bias)
        idx_rnd   = np.random.permutation(nF)
        rnd_seeds = samplesU['seeds'][j][:,idx_rnd]     # non-ordered seeds
           
        # sampling process using adaptive conditional sampling
        [u_j,geval,lam,_] = aCS(N,lam,b[j],rnd_seeds,H)
        
        # next level
        j = j+1   
        
        if b[j-1] <= 0 or j == max_it:
            break

    m = j
    samplesU['total'].append(u_j)  # store final failure samples (non-ordered)

    # delete unnecesary data
    if m < max_it:
        gsort = gsort[:m,:]
        prob  = prob[:m]
        b     = b[:m]

    ## acceptance probability and evidence
    p_acc = np.prod(prob)
    cE    = p_acc/c

    ## transform the samples to the physical (original) space
    for i in range(m+1):
        #p = dist_p.icdf(scipy.stats.normal.cdf(samplesU['total'][i][-1,:])) is the same as:
        p = scipy.stats.norm.cdf(samplesU['total'][i][-1,:])
        samplesX['total'].append(np.concatenate((T_nataf.U2X(samplesU['total'][i][:-1,:]), p.reshape(1,-1)), axis=0))
    
    return [b,samplesU,samplesX,cE]
##END
