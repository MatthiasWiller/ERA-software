import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from CEIS_SG import CEIS_SG
from CEIS_GM import CEIS_GM
from CEIS_vMFNM import CEIS_vMFNM
"""
---------------------------------------------------------------------------
Cross entropy method: Ex. 1 Ref. 2 - linear function of independent standard normal
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Comments:
* The CE-method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
* General convergence issues can be observed with linear LSFs.
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling 
   using Gaussian densities revisited"
   Geyer et al.
   Engineering Risk Analysis Group, TUM (Sep 2017)
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""

# %% definition of the random variables
d      = 2          # number of dimensions
pi_pdf = list()
for i in range(d):
    pi_pdf.append(ERADist('standardnormal', 'PAR', np.nan)) # n independent rv

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

# %% limit-state function
beta = 3.5
g    = lambda x: -np.sum(x, axis=0)/np.sqrt(d) + beta

# %% CE-method
N      = 1000        # Total number of samples for each level
rho    = 0.1         # Cross-correlation coefficient for conditional sampling
k_init = 3           # Initial number of distributions in the Mixture Model (GM/vMFNM)

print('CE-based IS stage: ')
# [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_SG(N, rho, g, pi_pdf)               # single gaussian
[Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_GM(N, rho, g, pi_pdf, k_init)       # gaussian mixture
# [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_vMFNM(N, rho, g, pi_pdf, k_init)    # adaptive vMFN mixture

# exact solution
pf_ex = sp.stats.norm.cdf(-beta)

# show p_f results
print('***Exact Pf: ', pf_ex, ' ***')
print('***CEIS Pf: ', Pr, ' ***\n\n')

# %% Plots
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

# %%END
