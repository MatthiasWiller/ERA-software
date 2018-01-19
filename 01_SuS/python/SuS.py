import numpy as np
from ERANataf import ERANataf
from ERADist import ERADist
from aCS import aCS
from MMA import MMA
from corr_factor import corr_factor
"""
---------------------------------------------------------------------------
Subset Simulation function (standard Gaussian space)
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-01
---------------------------------------------------------------------------
Input:
* N         : Number of samples per level
* p0        : Conditional probability of each subset
* g_fun     : limit state function
* distr     : Nataf distribution object or
              marginal distribution object of the input variables
---------------------------------------------------------------------------
Output:
* Pf_SuS    : failure probability estimate of subset simulation
* delta_SuS : coefficient of variation estimate of subset simulation 
* b         : intermediate failure levels
* Pf        : intermediate failure probabilities
* b_line    : limit state function values
* Pf_line   : failure probabilities corresponding to b_line
* samplesU  : samples in the Gaussian standard space for each level
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2. "MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""
def SuS(N,p0,g_fun,distr):
    ## initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        n = len(distr.Marginals)    # number of random variables (dimension)
        g = lambda u: g_fun(distr.U2X(u))   # LSF in standard space

        # if the samples are standard normal do not make any transform
        if distr.Marginals[0].Name.lower() == 'standardnormal':
            g = g_fun

    else:   # use distribution information for the transformation (independence)
        n   = len(distr)                    # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(normcdf(u))   # from u to x
        g   = lambda u: g_fun(u2x(u))               # LSF in standard space
        
        # if the samples are standard normal do not make any transform
        if distr[0].Name.lower() =='standardnormal':
            g = g_fun

    ## Initialization of variables and storage
    j      = 0                # initial conditional level
    Nc     = int(N*p0)        # number of markov chains
    Ns     = int(1/p0)        # number of samples simulated from each Markov chain
    lam    = 0.6              # recommended initial value for lambda
    max_it = 20               # estimated number of iterations
    samplesU = {'seeds': list(),
                'order': list()}
    #
    geval = np.zeros((N))        # space for the LSF evaluations
    gsort = np.zeros((max_it,N))   # space for the sorted LSF evaluations
    delta = np.zeros((max_it,1))   # space for the coefficient of variation
    Nf    = np.zeros((max_it,1))   # space for the number of failure point per level
    prob  = np.zeros((max_it,1))   # space for the failure probability at each level
    b     = np.zeros((max_it,1))   # space for the intermediate leveles

    ## SuS procedure
    # initial MCS stage
    print('Evaluating performance function:\t', end='')
    u_j = np.random.normal(size=(n,N))     # samples in the standard space
    for i in range(N):
        geval[i] = g(u_j[:,i])
        if geval[i] <= 0:
            Nf[j] = Nf[j]+1    # number of failure points
    print('OK! ')

    # SuS stage
    while True:
        # sort values in ascending order
        g_prime = np.sort(geval)
        gsort[j,:] = g_prime
        idx = sorted(range(len(geval)), key=lambda x: geval[x])
        
        # order the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU['order'].append(u_j_sort)   # store the ordered samples

        # intermediate level
        b[j] = np.percentile(gsort[j,:],p0*100)
        
        # number of failure points in the next level
        nF = sum(gsort[j,:] <= max(b[j],0))

        # assign conditional probability to the level
        if b[j] <= 0:
            b[j]    = 0
            prob[j] = nF/N
        else:
            prob[j] = p0      
        print('\n-Threshold intermediate level ', j, ' = ', b[j])

        # compute coefficient of variation
        if j == 0:
            delta[j] = np.sqrt(((1-p0)/(N*p0)))   # cov for p(1): MCS (Ref. 2 Eq. 8)
        else:
            I_Fj     = np.reshape(geval <= b[j],(Ns,Nc))       # indicator function for the failure samples
            p_j      = (1/N)*np.sum(I_Fj[:])                   # ~=p0, sample conditional probability
            gamma    = corr_factor(I_Fj,p_j,Ns,Nc)             # corr factor (Ref. 2 Eq. 10)
            delta[j] = np.sqrt( ((1-p_j)/(N*p_j))*(1+gamma) )  # coeff of variation(Ref. 2 Eq. 9)
        
        # select seeds
        samplesU['seeds'].append(u_j_sort[:,:nF])       # store ordered level seeds
        
        # randomize the ordering of the samples (to avoid bias)
        idx_rnd   = np.random.permutation(nF)
        rnd_seeds = samplesU['seeds'][j][:,idx_rnd]     # non-ordered seeds
        
        # sampling process using adaptive conditional sampling
        # [u_j,geval,lam,accrate] = aCS(N,lam,b[j],rnd_seeds,g)
        [u_j,geval,] = MMA(rnd_seeds,g,b[j],N)
        
        # next level
        j = j+1   
        
        if b[j-1] <= 0 or j-1 == max_it:
            break
        
    m = j
    samplesU['order'].append(u_j)  # store final failure samples (non-ordered)

    # delete unnecesary data
    if m < max_it:
        gsort = gsort[:m,:]
        prob  = prob[:m]
        b     = b[:m]
        delta = delta[:m]

    ## probability of failure
    # failure probability estimate
    Pf_SuS = np.prod(prob)   # or p0^(m-1)*(Nf(m)/N)

    # coefficient of variation estimate
    delta_SuS = np.sqrt(np.sum(delta**2))   # (Ref. 2 Eq. 12)

    ## Pf evolution 
    Pf           = np.zeros(m)
    Pf_line      = np.zeros((m,Nc))
    b_line       = np.zeros((m,Nc))
    Pf[0]        = p0
    Pf_line[0,:] = np.linspace(p0,1,Nc)
    b_line[0,:]  = np.percentile(gsort[0,:],Pf_line[0,:]*100)
    for i in range(1,m):
        Pf[i]        = Pf[i-1]*p0
        Pf_line[i,:] = Pf_line[i-1,:]*p0
        b_line[i,:]  = np.percentile(gsort[i,:],Pf_line[0,:]*100)

    Pf_line = np.sort(Pf_line.reshape(-1))
    b_line  = np.sort(b_line.reshape(-1))

    return Pf_SuS,delta_SuS,b,Pf,b_line,Pf_line,samplesU
##END