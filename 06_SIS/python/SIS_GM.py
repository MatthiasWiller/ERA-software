import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from h_calc import h_calc
from GM_sample import GM_sample
from EMGM import EMGM
"""
---------------------------------------------------------------------------
Sequential importance sampling
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
* N         : Number of samples per level
* rho       : 
* g_fun     : limit state function
* distr     : Nataf distribution object or
              marginal distribution object of the input variables
---------------------------------------------------------------------------
Output:

---------------------------------------------------------------------------
Based on:

---------------------------------------------------------------------------
"""
def SIS_GM(N, rho, g_fun, distr):

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
    max_it   = 100              # estimated number of iterations
    m        = 0                # counter for number of levels

    # Properties of SIS
    nsamlev  = N                # number of samples
    nchain   = nsamlev*rho      # number Markov chains
    lenchain = nsamlev/nchain   # number of samples per Markov chain
    burn     = 0                # burn-in

    # tolerance of COV of weights
    tolCOV = 1.5

    # initialize samples
    uk = np.zeros((nsamlev,dim))     # space for samples
    gk = np.zeros((1,nsamlev))       # space for evaluations of g
    accrate = np.zeros(max_it)   # space for acceptance rate
    Sk      = np.ones(max_it)    # space for expected weights

    ### Step 1
    # Perform the first Monte Carlo simulation
    for k in range(samlev):
        u       = randn((dim,1))
        uk[k,:] = u
        gk[k]   = g[u.T]

    # save samples 
    samplesU.append(uk.T)

    # set initial subset and failure level
    gmu         = np.mean(gk)
    sigmak[m] = 50*gmu

    ## Iteration
    for m in range(max_it):
      
        ### Step 2 and 3
        # compute sigma and weights
        if m == 1:
            sigma2    = fminbnd(@(x)abs(np.std(normcdf(-gk/x))/np.mean(normcdf(-gk/x))-tolCOV),0,10.0*gmu)
            sigmak[m] = sigma2
            wk        = normcdf(-gk/sigmak[m])
        else:     
            sigma2    = fminbnd(@(x)abs(np.std(normcdf(-gk/x)./normcdf(-gk/sigmak[m]))/np.mean(normcdf(-gk/x)./normcdf(-gk/sigmak[m-1]))-tolCOV),0,sigmak[m-1])
            sigmak[m] = sigma2
            wk        = normcdf(-gk/sigmak[m])/normcdf(-gk/sigmak[m-1])

        ### Step 4
        # compute estimate of expected w
        Sk[m] = np.mean(wk)

        # Exit algorithm if no convergence is achieved
        if Sk[m] == 0:
            break
        
        # compute normalized weights
        wnork = wk/Sk[m]/nsamlev

        # fit Gaussian Mixture
        nGM = 2
        [mu, si, pi] = EMGM(uk.T,wnork.T,nGM)
        
        ### Step 5
        # resample
        ind = randsample(nsamlev,nchain,true,wnork)

        # seeds for chains
        gk0 = gk[ind]
        uk0 = uk[ind,:]

        ### Step 6
        # perform M-H
        count = 0

        # initialize chain acceptance rate
        alphak = np.zeros([nchain])    

        # delete previous samples
        gk = []
        uk = []
      
        for k in range(nchain):
          # set seed for chain
          u0 = uk0[k,:]
          g0 = gk0[k]

          for i in range(lenchain+burn):
              count = count+1
                    
              if i == burn:
                  count = count-burn
                    
              # get candidate sample from conditional normal distribution 
              indw  = randsample(len(pi),1,true,pi)
              ucand = mvnrnd(mu[:,indw],si[:,:,indw])
                    
              # Evaluate limit-state function              
              gcand = g(ucand)             

              # compute acceptance probability
              pdfn = 0
              pdfd = 0
              for ii = range(len(pi)):
                  pdfn = pdfn+pi[ii]*mvnpdf(u0.T,mu[:,ii],si[:,:,ii])
                  pdfd = pdfd+pi[ii]*mvnpdf(ucand.T,mu[:,ii],si[:,:,ii])
                    
              alpha     = min(1,normcdf(-gcand/sigmak[m])*np.prod(normpdf(ucand))*pdfn/normcdf(-g0/sigmak(m+1))/prod(normpdf(u0))/pdfd)
              alphak[k] = alphak[k]+alpha/(lenchain+burn)

              # check if sample is accepted
              uhelp = sp.stats.uniform.rvs()
              if uhelp <= alpha:
                  uk[count,:] = ucand
                  gk[count]   = gcand
                  u0 = ucand
                  g0 = gcand
              else:
                  uk[count,:] = u0
                  gk[count]   = g0
                                    
        uk = uk[:nsamlev,:]
        gk = gk[:nsamlev]
        
        # save samples 
        samplesU.append(uk.T)

        # compute mean acceptance rate of all chains in level m
        accrate[m] = np.mean(alphak)
        COV_Sl = np.std((gk < 0)./normcdf(-gk/sigmak[m]))/np.mean((gk < 0)./normcdf(-gk/sigmak(m+1)))
        print('COV_Sl =',COV_Sl)
        if COV_Sl < 0.01:
            break

    # needed steps
    k_fin = len(pi) 
    l_tot  = m+1

    ## Calculation of the Probability of failure
    # accfin = accrate[m]
    const  = np.prod(Sk)
    Pr     = np.mean((gk < 0)./normcdf(-gk/sigmak[m]))*const

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