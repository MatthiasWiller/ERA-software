#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:45:40 2018

@author: felipe
"""

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from aBUS_SuS import aBUS_SuS
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True
plt.close('all')

# %% prior
n = 1   # number of random variables (dimensions)

# assign data: 1st variable is normal
mu_x    = 0
sigma_x = 1
dist_x1 = ERADist('normal','PAR',[mu_x, sigma_x])

# distributions
dist_X = [dist_x1]

# correlation matrix
R = np.eye(n)   # independent case

# object with distribution information
T_nataf = ERANataf(dist_X,R)

# %% likelihood
y_tilde    = 2
mu_nu      = 0
sigma_nu   = 0.5
likelihood = lambda x: sp.stats.norm.pdf(y_tilde-x, loc=mu_nu, scale=sigma_nu)
log_likelihood = lambda x: np.log(likelihood(x))

# %% c constant
c     = 0.5*np.sqrt(2*np.pi)
c_hat = 1/c

# %% BUS-SuS
N  = 2000       # number of samples per level
p0 = 0.1        # probability of each subset

# run the BUS_SuS.m function
[b,samplesU,samplesX,cE,c,lam] = aBUS_SuS(N,p0,log_likelihood,T_nataf)
       
# %% results
mu_exact    = 1.6   # if sigma_nu = 0.5
sigma_exact = 0.45  # if sigma_nu = 0.5
mu_xp       = np.mean(samplesX[-1][0,:])
sigma_xp    = np.std(samplesX[-1][0,:])
print('\nExact mean',mu_exact,'\nExact std:',sigma_exact,'\n')
print('Sample mean',mu_xp,'\nSample std:',sigma_xp,'\n')

# %% END