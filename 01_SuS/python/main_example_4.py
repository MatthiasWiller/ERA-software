import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
"""
---------------------------------------------------------------------------
Subset Simulation: Ex. 2 Ref. 3 - parabolic/concave limit-state function
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
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
*Express the failure probability as a product of larger conditional failure
 probabilities by introducing intermediate failure events.
*Use MCMC based on the modified Metropolis-Hastings algorithm for the 
 estimation of conditional probabilities.
*p0, the prob of each subset, is chosen 'adaptively' to be in [0.1,0.3]
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
3."Sequential importance sampling for structural reliability analysis"
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
b     = 5.0
kappa = 0.5
e     = 0.1
g     = lambda u: b - u[1,:] - kappa*(u[0,:]-e)**2

# %% subset simulation
N   = 1000        # Total number of samples for each level
p0  = 0.1         # Probability of each subset, chosen adaptively
alg = 'acs'       # Sampling Algorithm (either 'acs' or 'mma')

print('SUBSET SIMULATION stage: ')
[Pf_SuS,delta_SuS,h,Pf,h_sus,pf_sus,samplesU,samplesX] = SuS(N,p0,g,pi_pdf,alg)

# reference solution
pf_ref   = 3.01e-3

# show p_f results
print('\n***Reference Pf: ', pf_ref, ' ***')
print('***SuS Pf: ', Pf_SuS, ' ***\n\n')

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
    m = len(Pf)
    plt.figure() 
    xx = np.linspace(-6,6,240)
    nnp = len(xx) 
    [X,Y] = np.meshgrid(xx,xx)
    xnod = np.array([X,Y])
    Z    = g(xnod) 
    plt.contour(X,Y,Z,[0],colors='r')  # LSF
    for j in range(m+1):
        u_j_samples = samplesU['order'][j]
        plt.scatter(u_j_samples[0,:],u_j_samples[1,:],marker='.')
    
    plt.tight_layout()


# Plot failure probability: Exact
plt.figure()
plt.yscale('log')
plt.title('Failure probability estimate')
plt.xlabel('Limit state function, $g$')
plt.ylabel('Failure probability, $P_f$')

# Plot failure probability: SuS
plt.plot(h_sus,pf_sus,'r--', label='SuS')           # curve
plt.plot(h,Pf,'ko', label='Intermediate levels', 
                    markersize=8, 
                    markerfacecolor='none')         # points
plt.plot(0,Pf_SuS,'b+', label='Pf SuS', 
                        markersize=10, 
                        markerfacecolor='none')
plt.plot(0,pf_ref,'ro', label='Pf Ref.', 
                       markersize=10, 
                       markerfacecolor='none')
plt.legend()
plt.tight_layout()

plt.show()
# %%END
