import numpy as np
import scipy.stats
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
"""
---------------------------------------------------------------------------
Cross entropy method: Ex. 2 Ref. 2 - linear function of independent exponential
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
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

## Subset simulation
N  = 1000         # Total number of samples for each level
p0 = 0.1          # Probability of each subset, chosen adaptively

print('SUBSET SIMULATION stage: ')
[Pf_SuS,delta_SuS,b,Pf,b_sus,pf_sus,u_samples] = CEIS_SG(N,p0,g,pi_pdf)

# exact solution
lam      = 1
# pf_ex    = 1 - gamcdf(Ca,d,lam)
pf_ex    = 1 - scipy.stats.gamma.cdf(a=Ca, scale=lam, size=d)
# Pf_exact = lambda gg: 1-gamcdf(Ca-gg,d,lam)
Pf_exact = lambda gg: 1-scipy.stats.gamma.cdf(a=Ca-gg, scale=lam)
gg       = np.linspace(0,30,300)

# show p_f results
print('\n***Exact Pf: #g ***', pf_ex)
print('\n***SuS Pf: #g ***\n\n', Pf_SuS)

## Plots
# Plot failure probability: Exact
plt.figure()
plt.yscale('log')
plt.plot(gg,Pf_exact(gg),'b-', label='Exact')
plt.title('Failure probability estimate')
plt.xlabel('Limit state function, $g$')
plt.ylabel('Failure probability, $P_f$')

# Plot failure probability: SuS
plt.plot(b_sus,pf_sus,'r--', label='SuS')           # curve
plt.plot(b,Pf,'ko', label='Intermediate levels', 
                    markersize=8, 
                    markerfacecolor='none')         # points
plt.plot(0,Pf_SuS,'b+', label='Pf SuS', 
                        markersize=10, 
                        markerfacecolor='none')
plt.plot(0,pf_ex,'ro', label='Pf Exact', 
                       markersize=10, 
                       markerfacecolor='none')
plt.tight_layout()

plt.show()
##END
