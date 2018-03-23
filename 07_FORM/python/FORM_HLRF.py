import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
"""
---------------------------------------------------------------------------
HLRF function
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
* DG    : gradient of the limit state function in the standard space
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
def FORM_HLRF(G,DG,distr):

    # %% initial check if there exists a Nataf object
    if not(isinstance(distr, ERANataf)): 
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')

    n = len(distr.Marginals)    # number of random variables (dimension)

    # %% initialization
    maxit = int(1e2)
    tol   = 1e-5
    u     = np.zeros([n,maxit])
    beta  = np.zeros(maxit)

    # %% HLRF method
    for k in range(maxit):
        # 0. Get x and Jacobi from u (important for transformation)
        [xk, J] = distr.U2X(u[:,k].reshape(-1,1), Jacobian=True)
        
        # 1. evaluate LSF at point u_k
        H_uk = G(xk)
        
        # 2. evaluate LSF gradient at point u_k and direction cosines
        DH_uk      = sp.linalg.solve(J,DG(xk))
        norm_DH_uk = np.linalg.norm(DH_uk)
        alpha      = DH_uk/norm_DH_uk
        
        # 3. calculate beta
        beta[k] = -np.inner(u[:,k].T, alpha) + H_uk/norm_DH_uk
        
        # 4. calculate u_{k+1}
        u[:,k+1] = -beta[k]*alpha
        
        # next iteration
        if (np.linalg.norm(u[:,k+1]-u[:,k]) <= tol):
            break


    # delete unnecessary data
    u = u[:,:k+1]

    # compute design point, reliability index and Pf
    u_star = u[:,-1]
    x_star = distr.U2X(u_star.reshape(-1,1))
    beta   = beta[k]
    Pf     = sp.stats.norm.cdf(-beta)

    # print results
    print('*FORM Method\n')
    print(' ', k, ' iterations... Reliability index = ', beta, ' --- Failure probability = ', Pf, '\n\n')

    return [u_star, x_star, beta, Pf]