import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
"""
---------------------------------------------------------------------------
Optimization using the fmincon function (scipy.optimize.minimize() in Python)
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
Input:
* g     : limit state function in the original space
* distr : ERANataf-Object containing the distribution
---------------------------------------------------------------------------
Output:
* u_star : design point in the standard space
* x_star : design point in the original space
* beta   : reliability index
* Pf     : probability of failure
---------------------------------------------------------------------------
References:
1."Structural reliability"
   Lemaire et al. (2009)
   Wiley-ISTE.
---------------------------------------------------------------------------
"""
def FORM_fmincon(g, distr):
    
    # %% initial check if there exists a Nataf object
    if not(isinstance(distr, ERANataf)): 
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')

    d = len(distr.Marginals)

    # %% objective function
    dist_fun = lambda u: np.linalg.norm(u)

    # %% parameters of the minimize function
    u0  = np.tile([0.01],[d,1])  # initial search point

    # nonlinear constraint: H(u) <= 0
    H    = lambda u: g(distr.U2X(u.reshape(-1,1)))
    cons = ({'type': 'ineq', 'fun': lambda u: -H(u)})

    # method for minimization
    alg = 'SLSQP'
    
    # %% use constraint minimization
    res = sp.optimize.minimize(dist_fun, u0, constraints=cons, method=alg)

    # unpack results
    u_star = res.x
    beta   = res.fun
    it     = res.nit
    
    # compute design point in orignal space and failure probability
    x_star = distr.U2X(u_star.reshape(-1,1))
    Pf     = sp.stats.norm.cdf(-beta)
    # print results
    print('*scipy.optimize.minimize() with ', alg, ' Method\n')
    print(' ', it, ' iterations... Reliability index = ', beta, ' --- Failure probability = ', Pf, '\n\n')

    return [u_star, x_star, beta, Pf]