import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SIS_GM import SIS_GM
"""
---------------------------------------------------------------------------
Sequential importance sampling method: Ex. 2 Ref. 2 - linear function of independent exponential
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

---------------------------------------------------------------------------

---------------------------------------------------------------------------
"""

## definition of the random variables
d      = 100         # number of dimensions
pi_pdf = list()
for i in range(d):
    pi_pdf.append(ERADist('exponential', 'PAR',1)) # n independent rv

# correlation matrix
# R = eye(n)   # independent case

# object with distribution information
# pi_pdf = ERANataf(pi_pdf,R)    # if you want to include dependence

## limit-state function
Ca = 140
g  = lambda x: Ca - np.sum(x)

## CE-method
N   = 1000         # Total number of samples for each level
rho = 0.1          # Probability of each subset, chosen adaptively

print('SIS stage: ')
[Pr, l, samplesU, samplesX, k_fin] = SIS_GM(N, rho, g, pi_pdf)

# exact solution
lam      = 1
pf_ex    = 1 - sp.stats.gamma.cdf(a=Ca, scale=lam, size=d)
Pf_exact = lambda gg: 1-sp.stats.gamma.cdf(a=Ca-gg, scale=lam)
gg       = np.linspace(0,30,300)

# show p_f results
print('\n***Exact Pf: #g ***', pf_ex)
print('\n***CEIS Pf: #g ***\n\n', Pr)

##END
