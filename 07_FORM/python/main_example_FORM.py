import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_HLRF import FORM_HLRF
from FORM_fmincon import FORM_fmincon
"""
---------------------------------------------------------------------------
FORM example: using HLRF algorithm and fmincon
---------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
References:
1."Structural reliability"
   Lemaire et al. (2009)
   Wiley-ISTE.
2."Lecture Notes in Structural Reliability"
   Straub (2016)
---------------------------------------------------------------------------
"""

# %% definition of the random variables
d      = 2  # number of dimensions/random variables
pi_pdf = list()
pi_pdf.append(ERADist('normal','PAR',[1,0.4]))   # x1 ~ normpdf(1,0.4)
pi_pdf.append(ERADist('exponential','PAR',[5]))  # x2 ~ exppdf(5) --mu = 1/L = 0.2 --std = 1/L = 0.2

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf,R)    # if you want to include dependence

# %% limit state function and its gradient in the original space
G  = lambda x: x[0] - x[1]
DG = lambda x: np.array([1, -1])

# %% Solve the optimization problem of the First Order Reliability Method

# OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf] = FORM_HLRF(G, DG, pi_pdf)

# OPC 2. FORM using MATLAB fmincon
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc] = FORM_fmincon(G, pi_pdf)

# %% plots HLRF results
G = lambda x1, x2: x1 - x2
H = lambda u1, u2: (2*u1)/5 + np.log(1/2 - sp.special.erf((2**(.5)*u2)/2)/2)/5 + 1

# grid points
xx      = np.linspace(-1,3,50)
[X1,X2] = np.meshgrid(xx,xx)
uu1     = np.linspace(-3,1,50)
uu2     = np.linspace(-1,3,50)
[U1,U2] = np.meshgrid(uu1,uu2)

# figure
f, (ax1, ax2) = plt.subplots(1, 2)
#
ax1.pcolor(X1,X2,G(X1,X2))
ax1.contour(X1,X2,G(X1,X2),[0]) 
ax1.plot(x_star_hlrf[0],x_star_hlrf[1],'r*')   # design point in original
ax1.set_title('Physical space')
ax1.set_aspect('equal', 'box')
#
ax2.pcolor(U1,U2,G(U1,U2))
ax2.contour(U1,U2,H(U1,U2),[0]) 
ax2.plot([0, u_star_hlrf[0]],[0, u_star_hlrf[1]])   # reliability index beta
ax2.plot(u_star_hlrf[0],u_star_hlrf[1],'r*')        # design point in standard
ax2.plot(0,0,'ro')                                  # origin in standard
ax2.set_title('Standard space')
ax2.set_aspect('equal', 'box')

plt.show()

# %%END