import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from aBUS_SuS import aBUS_SuS
"""
---------------------------------------------------------------------------
aBUS+SuS: Ex. 1 Ref. 1 - Parameter identification 12D example
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
References:
1."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
"""

# %% definition of the random variables
n = 12   # number of random variables (dimensions)
# assign data: 1st variable is Lognormal
dist_x = ERADist('standardnormal','PAR',[0,1])

# distributions
dist_X = list()
for i in range(n):
    dist_X.append(dist_x)
# correlation matrix
R = np.eye(n)   # independent case

# object with distribution information
T_nataf = ERANataf(dist_X,R)

# %% likelihood function
mu_l           = 0.462
sigma_l        = 0.6
likelihood     = lambda u: np.prod(sp.stats.norm.pdf((u-mu_l)/sigma_l), axis=0)/sigma_l
realmin        = np.finfo(np.double).tiny # realmin to avoid Inf values in log(0)
log_likelihood = lambda u: np.log(likelihood(u) + realmin)

# %% aBUS-SuS
N  = 2000       # number of samples per level
p0 = 0.1        # probability of each subset

# run the BUS_SuS.m function
[h,samplesU,samplesX,cE,c,lam_new] = aBUS_SuS(N,p0,log_likelihood,T_nataf)

# %% organize samples and show results
nsub = len(h.flatten())+1   # number of levels + final posterior
u1p = list()
u0p = list()
x1p = list()
pp  = list()

for i in range(nsub):
   # samples in standard
   u1p.append(samplesU['total'][i][0,:])  
   u0p.append(samplesU['total'][i][-1,:])
   # samples in physical
   x1p.append(samplesX[i][0,:])
   pp.append( samplesX[i][-1,:])

# reference solutions
mu_exact    = 0.34     # for all x_i
sigma_exact = 0.51     # for all x_i
cE_exact    = 1.0e-6

# print results
print('\nExact model evidence =', cE_exact)
print('Model evidence BUS-SuS =', cE, '\n')
print('Exact posterior mean x_1 =', mu_exact)
print('Mean value of x_1 =', np.mean(x1p[-1]), '\n')
print('Exact posterior std x_1 =', sigma_exact)
print('Std of x_1 =', np.std(x1p[-1]))
# %%END
