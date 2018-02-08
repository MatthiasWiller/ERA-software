import numpy as np
import scipy
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from iTMCMC import iTMCMC
from shear_building_2DOF import shear_building_2DOF
"""
---------------------------------------------------------------------------
iTMCMC: Ex. 1 Ref. 1 - Parameter identification two-DOF shear building
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-01
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements". 
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Transitional Markov Chain Monte Carlo method for Bayesian model updating, 
   model class selection and model averaging". 
   Jianye Ching & Yi-Chun Chen
   Journal of Engineering Mechanics. 133.7 (2007) 816-832
---------------------------------------------------------------------------
"""

## model data
# shear building data
m1 = 16.5e3     # mass 1st story [kg]
m2 = 16.1e3     # mass 2nd story [kg]
kn = 29.7e6     # nominal values for the interstory stiffnesses [N/m]

## prior PDF for X1 and X2 (product of lognormals)
mod_log_X1 = 1.3   # mode of the lognormal 1
std_log_X1 = 1.0   # std of the lognormal 1
mod_log_X2 = 0.8   # mode of the lognormal 2
std_log_X2 = 1.0   # std of the lognormal 2

# find lognormal X1 parameters
var_fun = lambda mu: std_log_X1**2 - (np.exp(mu-np.log(mod_log_X1))-1) \
                                     *np.exp(2*mu+(mu-np.log(mod_log_X1)))
mu_X1  = scipy.optimize.fsolve(var_fun,1)   # mean of the associated Gaussian
std_X1 = np.sqrt(mu_X1-np.log(mod_log_X1))  # std of the associated Gaussian

# find lognormal X2 parameters
var_X2 = lambda mu: std_log_X2**2 - (np.exp(mu-np.log(mod_log_X2))-1) \
                                    *np.exp(2*mu+(mu-np.log(mod_log_X2)))
mu_X2  = scipy.optimize.fsolve(var_X2,0)     # mean of the associated Gaussian
std_X2 = np.sqrt(mu_X2-np.log(mod_log_X2))   # std of the associated Gaussian

## definition of the random variables
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

## likelihood function
lam     = np.array([1, 1])   # means of the prediction error
i       = 9                  # simulation level
var_eps = 0.5**(i-1)         # variance of the prediction error
f_tilde = np.array([3.13, 9.83])       # measured eigenfrequencies [Hz]

# shear building model 
f = lambda x: shear_building_2DOF(m1, m2, kn*x[0], kn*x[1])

# modal measure-of-fit function
J = lambda x: np.sum((lam**2)*(((f(x)**2)/f_tilde**2) - 1)**2)   

# likelihood function
likelihood     = lambda x: np.exp(-J(x)/(2*var_eps))
realmin        = np.finfo(np.double).tiny # realmin to avoid Inf values in log(0)
log_likelihood = lambda x: np.log(np.exp(-J(x)/(2*var_eps)) + realmin)


## TMCMC
Ns = int(1e3)    # number of samples per level
Nb = int(0.1*Ns) # burn-in period

# run the iTMCMC.py function
[Theta,p,S] = iTMCMC(Ns, Nb, log_likelihood, T_nataf)

## show results
# reference solutions
mu_exact    = 1.12   # for x_1
sigma_exact = 0.66   # for x_1
cE_exact    = 1.52e-3

print('Exact model evidence =', cE_exact)
print('Model evidence TMCMC =', S, '\n')
print('Exact posterior mean x_1 =', mu_exact)
print('Mean value of x_1 =', np.mean(Theta['original'][-1][:,0]), '\n')
print('Exact posterior std x_1 = ', sigma_exact)
print('Std of x_1 =',np.std(Theta['original'][-1][:,0]),'\n\n')

## Plots
# Options
# ...
m = len(p)   # number of stages (intermediate levels)
# plot p values
plt.figure()
plt.plot(np.range(1,m),p,'ro-')
plt.xlabel('Intermediate levels $j$') # ,'Interpreter','Latex','FontSize', 18)
plt.ylabel('$p_j$') #,'Interpreter','Latex','FontSize', 18)
# set(gca,'FontSize',15) axis tight
   
# plot samples increasing p
idx = np.array([0, np.round((m-1)/3), np.round(2*(m-1)/3), m-1])
plt.figure()
for i in range(4):
   plt.subplot(2,2,i+1) 
   plt.plot(Theta['original'][idx[i]][:,0],Theta['original'][idx[i]][:,1],'b.')
   plt.title('$p_j=',p[idx[i]],'$') #,'Interpreter','Latex','FontSize', 18)
   plt.xlabel('$\theta_1$') # ,'Interpreter','Latex','FontSize', 18)
   plt.ylabel('$\theta_2$') # ,'Interpreter','Latex','FontSize', 18)
   plt.xlim([0, 3])
   plt.ylim([0, 2])
   #set(gca,'FontSize',15)  axis equal 


plt.show()
##END