import numpy as np
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from BUS_SuS import BUS_SuS
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
var_fun = lambda mu: std_log_X1^2 - (np.exp(mu-np.log(mod_log_X1))-1) \
                                    *np.exp(2*mu+(mu-np.log(mod_log_X1)))
mu_X1  = np.fzero(var_fun,0)                # mean of the associated Gaussian
std_X1 = np.sqrt(mu_X1-np.log(mod_log_X1))  # std of the associated Gaussian

# find lognormal X2 parameters
var_X2 = lambda mu: std_log_X2^2 - (np.exp(mu-np.log(mod_log_X2))-1) \
                                   *np.exp(2*mu+(mu-np.log(mod_log_X2)))
mu_X2  = np.fzero(var_X2,0)                 # mean of the associated Gaussian
std_X2 = np.sqrt(mu_X2-np.log(mod_log_X2))  # std of the associated Gaussian

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
lam     = [1, 1]             # means of the prediction error
i       = 9                  # simulation level
var_eps = 0.5**(i-1)         # variance of the prediction error
f_tilde = [3.13, 9.83]       # measured eigenfrequencies [Hz]

# shear building model 
f = lambda x: shear_building_2DOF(m1, m2, kn*x(1), kn*x(2))

# modal measure-of-fit function
J = lambda x: sum((lam**2)*(((f(x)**2)/f_tilde**2) - 1)**2)   

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
    K  = 5e3                            # number of samples      
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
    error('Finding the scale constant c requires -method- 1, 2 or 3')

## BUS-SuS
N  = 2000       # number of samples per level
p0 = 0.1        # probability of each subset

# run the BUS_SuS.m function
[b,samplesU,samplesX,cE] = BUS_SuS(N,p0,c,likelihood,T_nataf)

## organize samples and show results
nsub = len(b)+1   # number of levels + final posterior
u1p  = cell(nsub,1)   u2p  = cell(nsub,1)   u0p = cell(nsub,1)
x1p  = cell(nsub,1)   x2p  = cell(nsub,1)   pp  = cell(nsub,1)
for i in range(nsub):
   # samples in standard
   u1p{i} = samplesU.total{i}(1,:)             
   u2p{i} = samplesU.total{i}(2,:)  
   u0p{i} = samplesU.total{i}(3,:) 
   # samples in physical
   x1p{i} = samplesX.total{i}(1,:)   
   x2p{i} = samplesX.total{i}(2,:) 
   pp{i}  = samplesX.total{i}(3,:)

# reference solutions
mu_exact    = 1.12     # for x_1
sigma_exact = 0.66     # for x_1
cE_exact    = 1.52e-3

# show results
fprintf('\nExact model evidence =', cE_exact)
fprintf('\nModel evidence BUS-SuS =', cE, '\n')
fprintf('\nExact posterior mean x_1 =', mu_exact)
fprintf('\nMean value of x_1 =', np.mean(x1p{end}), '\n')
fprintf('\nExact posterior std x_1 =', sigma_exact)
fprintf('\nStd of x_1 =', std(x1p{end}), '\n\n')

## plot samples
plt.figure()
for i in range(nsub):
   subplot(2,2,i) plot(u1p{i},u2p{i},'r.') 
   xlabel('$u_1$','Interpreter','Latex','FontSize', 18)   
   ylabel('$u_2$','Interpreter','Latex','FontSize', 18)
   set(gca,'FontSize',15) axis equal xlim([-3 1]) ylim([-3 0])

annotation('textbox', [0 0.9 1 0.1],'String', '\bf Standard space', ...
           'EdgeColor', 'none', 'HorizontalAlignment', 'center')

plt.figure()
for i in range(nsub):
   subplot(2,2,i) plot(x1p{i},x2p{i},'b.') 
   xlabel('$x_1$','Interpreter','Latex','FontSize', 18)
   ylabel('$x_2$','Interpreter','Latex','FontSize', 18)
   set(gca,'FontSize',15) axis equal xlim([0 3]) ylim([0 1.5])

annotation('textbox', [0 0.9 1 0.1],'String', '\bf Original space', ...
           'EdgeColor', 'none', 'HorizontalAlignment', 'center')

plt.show()
##END
