import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_HLRF import FORM_HLRF
from FORM_fmincon import FORM_fmincon
"""
---------------------------------------------------------------------------
FORM using HLRF algorithm and fmincon: Ex. 2 Ref. 3 - parabolic/concave limit-state function
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
Comment:
* The FORM method uses a first order approximation of the LSF and is 
  therefore not accurate for non-linear LSF's
---------------------------------------------------------------------------
Based on:
1."Structural reliability"
   Lemaire et al. (2009)
   Wiley-ISTE.
2."Lecture Notes in Structural Reliability"
   Straub (2016)
3. "Sequential importance sampling for structural reliability analysis"
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

# %% limit state function and its gradient in the original space
b     = 5.0
kappa = 0.5
e     = 0.1
g     = lambda u: b - u[1,:] - kappa*(u[0,:]-e)**2
dg    = lambda u: np.array([-2*kappa*(u[0,:]-e), np.array([-1.0])])

# %% Solve the optimization problem of the First Order Reliability Method

# OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf] = FORM_HLRF(g, dg, pi_pdf)

# OPC 2. FORM using Python scipy.optimize.minimize()
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc] = FORM_fmincon(g, pi_pdf)

# reference solution
pf_ref = 3.01e-3

# show p_f results
print('\n***Reference Pf: ', pf_ref, ' ***')
print('***FORM HLRF Pf: ', Pf_hlrf, ' ***')
print('***FORM fmincon Pf: ', Pf_fmc, ' ***\n\n')

# %% Plot HLRF results
if d == 2:
    # grid points
    xx      = np.linspace(-6,6,100)
    [X1,X2] = np.meshgrid(xx,xx)
    xnod    = np.array([X1,X2])
    ZX      = g(xnod)

    # figure
    plt.figure()
    #
    plt.pcolor(X1,X2,ZX)
    plt.contour(X1,X2,ZX,[0]) 
    plt.plot([0, x_star_hlrf[0]],[0, x_star_hlrf[1]])   # reliability index beta
    plt.plot(x_star_hlrf[0],x_star_hlrf[1],'r*')        # design point in standard
    plt.plot(0,0,'ro')                                  # origin in standard    plt.title('Standard space')
    plt.axes().set_aspect('equal', 'box')

    plt.show()

# %%END