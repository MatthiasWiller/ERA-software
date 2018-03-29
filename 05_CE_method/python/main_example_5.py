import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from CEIS_SG import CEIS_SG
from CEIS_GM import CEIS_GM
# from CEIS_vMFNM import CEIS_vMFNM
"""
---------------------------------------------------------------------------
Cross entropy method: Ex. 3 Ref. 1 - series system reliability problem
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling 
   using Gaussian densities revisited"
   Geyer et al.
   Engineering Risk Analysis Group, TUM (Sep 2017)
2. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75   
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
g = lambda u: np.amin(np.array([ 
                    0.1*(u[0,:]-u[1,:])**2-(u[0,:]+u[1,:])/np.sqrt(2)+3 , 
                    0.1*(u[0,:]-u[1,:])**2+(u[0,:]+u[1,:])/np.sqrt(2)+3 , 
                    u[0,:]-u[1,:]+7/np.sqrt(2) , 
                    u[1,:]-u[0,:]+7/np.sqrt(2) 
                    ]), axis=0)

# %% Cross entropy-based IS
N   = 5000        # Total number of samples for each level
rho = 0.1         # cross-correlation coefficient

print('CE-based IS stage: ')
# [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_SG(N, rho, g, pi_pdf)       # single gaussian
[Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_GM(N, rho, g, pi_pdf)       # gaussian mixture
# [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_vMFNM(N, rho, g, pi_pdf)    # adaptive vMFN mixture

# reference solution
pf_ref = 2.2e-3

# show p_f results
print('\n***Reference Pf: ', pf_ref, ' ***')
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
    plt.figure() 
    xx = np.linspace(-7,7,280)
    nnp = len(xx) 
    [X,Y] = np.meshgrid(xx,xx)
    xnod = np.array([X,Y])
    Z    = g(xnod) 
    plt.contour(X,Y,Z,[0],colors='r')  # LSF
    for j in range(l+1):
        u_j_samples = samplesU[j]
        plt.scatter(u_j_samples[0,:],u_j_samples[1,:],marker='.')
    
    plt.tight_layout()


plt.show()

# %% END
