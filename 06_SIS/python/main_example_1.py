import numpy as np
import scipy.stats
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SIS_GM import SIS_GM
"""
---------------------------------------------------------------------------
Sequential importance sampling method: Ex. 1 Ref. 2 - linear function of independent standard normal
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
Based on:

---------------------------------------------------------------------------
"""

## definition of the random variables
d      = 2          # number of dimensions
pi_pdf = list()
for i in range(d):
    pi_pdf.append(ERADist('standardnormal', 'PAR', np.nan)) # n independent rv

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

## limit-state function
beta = 3.5
g    = lambda x: -x.sum(axis=0)/np.sqrt(d) + beta

## cross entropy-based IS
N   = 1000        # Total number of samples for each level
rho = 0.1         # ...

print('SIS stage: ')
[Pr, l, samplesU, samplesX, k_fin] = SIS_GM(N, rho, g, pi_pdf)

# exact solution
pf_ex    = scipy.stats.norm.cdf(-beta)
Pf_exact = lambda gg: scipy.stats.norm.cdf(gg,beta,1)
gg       = np.linspace(0,7,140)

# show p_f results
print('***Exact Pf: ', pf_ex, ' ***')
print('***CEIS Pf: ', Pr, ' ***\n\n')

## Plots
# Options for font-family and font-size
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title
# Plot samples
if d == 2:
    m = l
    plt.figure() 
    xx = np.linspace(0,5,100)
    nnp = len(xx) 
    [X,Y] = np.meshgrid(xx,xx)
    xnod = np.array([X,Y])
    Z    = g(xnod) 
    plt.contour(X,Y,Z,[0],colors='r')  # LSF
    for j in range(m+1):
        u_j_samples = samplesU[j]
        plt.scatter(u_j_samples[0,:],u_j_samples[1,:],marker='.')
    
    plt.tight_layout()


plt.show()

##END
