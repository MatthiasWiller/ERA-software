import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
"""
---------------------------------------------------------------------------
Optimization using the fmincon function
---------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* G     : limit state function in the original space
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
def FORM_fmincon(G, distr):

    # %% objective function
    dist_fun = lambda u: np.linalg.norm(u)

    # %% parameters of the minimize function
    u0  = [0.1,0.1]  # initial search point

    # nonlinear constraint: H(u) <= 0
    H      = lambda u: G(distr.U2X(u.reshape(-1,1)))
    cons = ({'type': 'ineq', 'fun': lambda u: -H(u)})

    # boundaries of the optimization
    bnds = ((-5, 5), (-5, 5))

    # method for minimization
    alg = 'SLSQP'
    
    # %% use constraint minimization
    res = sp.optimize.minimize(dist_fun, u0, bounds=bnds, constraints=cons, method=alg)

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