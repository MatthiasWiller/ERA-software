import numpy as np
import scipy
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from BUS_SuS import BUS_SuS
from shear_building_2DOF import shear_building_2DOF
"""
---------------------------------------------------------------------------
BUS+SuS: Ex. 1 Ref. 1 - Parameter identification two-DOF shear building
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
1."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
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
likelihood = lambda x: np.exp(-J(x)/(2*var_eps))

## find scale constant c
method = 2

# use MLE to find c
if method == 1:
    f_start = np.log([mu_X1, mu_X2])
    fun     = lambda lnF: -np.log(likelihood(np.exp(lnF)) + realmin)
    # TODO: optimset, fminsearch in python !
    options = optimset('MaxIter',1e7,'MaxFunEvals',1e7)
    MLE_ln  = fminsearch(fun,f_start,options)
    MLE     = np.exp(MLE_ln)   # likelihood(MLE) = 1
    c       = 1/likelihood(MLE)
      
# some likelihood evaluations to find c
elif method == 2:
    K  = int(5e3)                       # number of samples      
    u  = np.random.normal(size=(n,K))   # samples in standard space
    x  = T_nataf.U2X(u)                 # samples in physical space
    # likelihood function evaluation
    L_eval = np.zeros((K,1))
    for i in range(K):
        L_eval[i] = likelihood(x[:,i])
    c = 1/np.max(L_eval)    # Ref. 1 Eq. 34
      
# use approximation to find c
elif method == 3:
    print('This method requires a large number of measurements')
    m = len(f_tilde)
    p = 0.05
    # TODO: chi2inv in python !
    c = 1/np.exp(-0.5*chi2inv(p,m))    # Ref. 1 Eq. 38
    
else:
    raise RuntimeError('Finding the scale constant c requires -method- 1, 2 or 3')

## BUS-SuS
N  = 2000       # number of samples per level
p0 = 0.1        # probability of each subset

# run the BUS_SuS.m function
[b,samplesU,samplesX,cE] = BUS_SuS(N,p0,c,likelihood,T_nataf)

## organize samples and show results
nsub = len(b.flatten())+1   # number of levels + final posterior
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
   x1p.append(samplesX['total'][i][0][0,:])
   x2p.append(samplesX['total'][i][0][1,:])
   pp.append( samplesX['total'][i][1])

# reference solutions
mu_exact    = 1.12     # for x_1
sigma_exact = 0.66     # for x_1
cE_exact    = 1.52e-3

# show results
print('Exact model evidence =', cE_exact)
print('Model evidence BUS-SuS =', cE, '\n')
print('Exact posterior mean x_1 =', mu_exact)
print('Mean value of x_1 =', np.mean(x1p[-1]), '\n')
print('Exact posterior std x_1 =', sigma_exact)
print('Std of x_1 =', np.std(x1p[-1]))

## Plots
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
   plt.subplot(2,2,i+1) 
   plt.plot(u1p[i],u2p[i],'r.') 
   plt.xlabel('$u_1$') 
   plt.ylabel('$u_2$')
   plt.xlim([-3, 1])
   plt.ylim([-3, 0])

# plot samples original space
plt.figure()
plt.suptitle('Original space')
for i in range(nsub):
   plt.subplot(2,2,i+1) 
   plt.plot(x1p[i],x2p[i],'b.') 
   plt.xlabel('$x_1$')
   plt.ylabel('$x_2$')
   plt.xlim([0, 3])
   plt.ylim([0, 1.5])

plt.show()
##END
