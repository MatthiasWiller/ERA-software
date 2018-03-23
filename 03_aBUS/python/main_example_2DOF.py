import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from aBUS_SuS import aBUS_SuS
from shear_building_2DOF import shear_building_2DOF
"""
---------------------------------------------------------------------------
aBUS+SuS: Ex. 1 Ref. 1 - Parameter identification two-DOF shear building
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

# %% model data
# shear building data
m1 = 16.5e3     # mass 1st story [kg]
m2 = 16.1e3     # mass 2nd story [kg]
kn = 29.7e6     # nominal values for the interstory stiffnesses [N/m]

# %% prior PDF for X1 and X2 (product of lognormals)
mod_log_X1 = 1.3   # mode of the lognormal 1
std_log_X1 = 1.0   # std of the lognormal 1
mod_log_X2 = 0.8   # mode of the lognormal 2
std_log_X2 = 1.0   # std of the lognormal 2

# find lognormal X1 parameters
var_fun = lambda mu: std_log_X1**2 - (np.exp(mu-np.log(mod_log_X1))-1) \
                                     *np.exp(2*mu+(mu-np.log(mod_log_X1)))
mu_X1  = sp.optimize.fsolve(var_fun,1)   # mean of the associated Gaussian
std_X1 = np.sqrt(mu_X1-np.log(mod_log_X1))  # std of the associated Gaussian

# find lognormal X2 parameters
var_X2 = lambda mu: std_log_X2**2 - (np.exp(mu-np.log(mod_log_X2))-1) \
                                    *np.exp(2*mu+(mu-np.log(mod_log_X2)))
mu_X2  = sp.optimize.fsolve(var_X2,0)     # mean of the associated Gaussian
std_X2 = np.sqrt(mu_X2-np.log(mod_log_X2))   # std of the associated Gaussian

# %% definition of the random variables
n = 2   # number of random variables (dimensions)
# assign data: 1st variable is Lognormal
dist_x1 = ERADist('lognormal','PAR',[mu_X1, std_X1])
# assign data: 2nd variable is Lognormal
dist_x2 = ERADist('lognormal','PAR',[mu_X2, std_X2])

# distributions
dist_X = [dist_x1, dist_x2]
# correlation matrix
R = np.eye(n)   # independent case

# object with distribution information
T_nataf = ERANataf(dist_X,R)

# %% likelihood function
lam     = np.array([1, 1])   # means of the prediction error
i       = 9                  # simulation level
var_eps = 0.5**(i-1)         # variance of the prediction error
f_tilde = np.array([3.13, 9.83])       # measured eigenfrequencies [Hz]

# shear building model 
f = lambda x: shear_building_2DOF(m1, m2, kn*x[0], kn*x[1])

# modal measure-of-fit function
J = lambda x: sum((lam**2)*(((f(x)**2)/f_tilde**2) - 1)**2)   

# likelihood function
likelihood     = lambda x: np.exp(-J(x)/(2*var_eps))
realmin        = np.finfo(np.double).tiny # realmin to avoid Inf values in log(0)
log_likelihood = lambda x: np.log(np.exp(-J(x)/(2*var_eps)) + realmin)

# %% aBUS-SuS
N  = 2000       # number of samples per level
p0 = 0.1        # probability of each subset

# run the BUS_SuS.m function
[h,samplesU,samplesX,cE,c,lam_new] = aBUS_SuS(N,p0,log_likelihood,T_nataf)

# %% organize samples and show results
nsub = len(h.flatten())+1   # number of levels + final posterior
u1p = list()
u2p = list()
u0p = list()
x1p = list()
x2p = list()
pp  = list()

for i in range(nsub):
   # samples in standard
   u1p.append(samplesU['total'][i][0,:])  
   u2p.append(samplesU['total'][i][1,:])
   u0p.append(samplesU['total'][i][2,:])
   # samples in physical
   x1p.append(samplesX[i][0,:])
   x2p.append(samplesX[i][1,:])
   pp.append( samplesX[i][2,:])

# reference solutions
mu_exact    = 1.12     # for x_1
sigma_exact = 0.66     # for x_1
cE_exact    = 1.52e-3

# show results
print('\nExact model evidence =', cE_exact)
print('Model evidence BUS-SuS =', cE, '\n')
print('Exact posterior mean x_1 =', mu_exact)
print('Mean value of x_1 =', np.mean(x1p[-1]), '\n')
print('Exact posterior std x_1 =', sigma_exact)
print('Std of x_1 =', np.std(x1p[-1]))

# %% Plots
# Options for font-family and font-size
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# plot samples standard space
plt.figure()
plt.suptitle('Standard space')
for i in range(nsub):
   plt.subplot(2,3,i+1) 
   plt.plot(u1p[i],u2p[i],'r.') 
   plt.xlabel('$u_1$') 
   plt.ylabel('$u_2$')
   plt.xlim([-3, 1])
   plt.ylim([-3, 0])

# plot samples original space
plt.figure()
plt.suptitle('Original space')
for i in range(nsub):
   plt.subplot(2,3,i+1) 
   plt.plot(x1p[i],x2p[i],'b.') 
   plt.xlabel('$x_1$') 
   plt.ylabel('$x_2$')
   plt.xlim([0, 3])
   plt.ylim([0, 1.5])

plt.show()
# %%END
