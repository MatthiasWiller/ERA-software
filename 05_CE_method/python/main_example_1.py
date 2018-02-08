import numpy as np
import scipy.stats
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from CEIS_SG import CEIS_SG
"""
---------------------------------------------------------------------------
Cross entropy method: Ex. 1 Ref. 2 - linear function of independent standard normal
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
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

print('CE-based IS stage: ')
[Pr, l, N_tot, gamma_hat, u_samples, k_fin] = CEIS_SG(N, rho, g, pi_pdf)       # single gaussian
# [Pr, l, N_tot, gamma_hat, k_fin] = CEIS_GM(N, rho, g, pi_pdf)       # gaussian mixture
# [Pr, l, N_tot, gamma_hat, k_fin] = CEIS_vMFNM(N, rho, g, pi_pdf)    # adaptive vMFN mixture



# exact solution
pf_ex    = scipy.stats.norm.cdf(-beta)
Pf_exact = lambda gg: scipy.stats.norm.cdf(gg,beta,1)
gg       = np.linspace(0,7,140)

# show p_f results
print('\n***Exact Pf: ', pf_ex, ' ***')
print('\n***CEIS Pf: ', Pr, ' ***\n\n')

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
        u_j_samples = u_samples['total'][j]
        plt.scatter(u_j_samples[0,:],u_j_samples[1,:],marker='.')
    
    plt.tight_layout()


# # Plot failure probability: Exact
# plt.figure()
# plt.yscale('log')
# plt.plot(gg,Pf_exact(gg),'b-', label='Exact')
# plt.title('Failure probability estimate')
# plt.xlabel('Limit state function, $g$')
# plt.ylabel('Failure probability, $P_f$')

# # Plot failure probability: SuS
# plt.plot(b_sus,pf_sus,'r--', label='SuS')           # curve
# plt.plot(b,Pf,'ko', label='Intermediate levels', 
#                     markersize=8, 
#                     markerfacecolor='none')         # points
# plt.plot(0,Pf_SuS,'b+', label='Pf SuS', 
#                         markersize=10, 
#                         markerfacecolor='none')
# plt.plot(0,pf_ex,'ro', label='Pf Exact', 
#                        markersize=10, 
#                        markerfacecolor='none')
# plt.tight_layout()

plt.show()
##END
