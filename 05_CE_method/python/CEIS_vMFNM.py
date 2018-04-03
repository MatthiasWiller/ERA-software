import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from h_calc import h_calc
from EMvMFNM import EMvMFNM
"""
---------------------------------------------------------------------------
Cross entropy-based importance sampling with vMFNM-distribution
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* N      : Number of samples per level
* rho    : cross-correlation coefficient for conditional sampling
* g_fun  : limit state function
* distr  : Nataf distribution object or
          marginal distribution object of the input variables
* k_init : initial number of distributions in the mixture model
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* l         : total number of levels
* N_tot     : total number of samples
* gamma_hat : gamma_ik probability of sample i belonging to distribution k 
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of distributions in the mixture
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling 
   using Gaussian densities revisited"
   Geyer et al.
   Engineering Risk Analysis Group, TUM (Sep 2017)
---------------------------------------------------------------------------
"""
def CEIS_vMFNM(N, rho, g_fun, distr, k_init):
    # %% initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        dim = len(distr.Marginals)    # number of random variables (dimension)
        g   = lambda u: g_fun(distr.U2X(u))   # LSF in standard space

        # if the samples are standard normal do not make any transform
        if distr.Marginals[0].Name.lower() == 'standardnormal':
            g = g_fun

    elif isinstance(distr, ERADist):   # use distribution information for the transformation (independence)
        dim = len(distr)                    # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))   # from u to x
        g   = lambda u: g_fun(u2x(u))                            # LSF in standard space
        
        # if the samples are standard normal do not make any transform
        if distr[0].Name.lower() =='standardnormal':
            g = g_fun
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')
    
    if dim < 2:
        raise RuntimeError('Sorry, the vMFN-model can only be applied to d > 1!')

    # %% Initialization of variables and storage
    j      = 0                # initial level
    max_it = 100              # estimated number of iterations
    N_tot  = 0                # total number of samples

    # Definition of parameters of the random variables (uncorrelated standard normal)
    gamma_hat = np.zeros([max_it+1]) # space for gamma
    samplesU = list()

    # %% CE procedure
    # initial nakagami parameters (make it equal to chi distribution)
    omega_init = dim    # spread parameter
    m_init     = dim/2  # shape parameter

    # initial von Mises-Fisher parameters
    kappa_init = 0                   # Concentration parameter (zero for uniform distribution)
    mu_init    = hs_sample(1,dim,1) # Initial mean sampled on unit hypersphere

    # initial disribution weight
    alpha_init = np.array([1.0])

    # %% Initializing parameters
    mu_hat       = mu_init
    kappa_hat    = kappa_init
    omega_hat    = omega_init
    m_hat        = m_init
    gamma_hat[j] = 1
    alpha_hat    = alpha_init
        
    # %% Iteration
    for j in range(max_it):
        # save parameters from previous step
        mu_cur    = mu_hat
        kappa_cur = kappa_hat
        omega_cur = omega_hat
        m_cur     = m_hat
        alpha_cur = alpha_hat

        # Generate samples
        X = vMFNM_sample(mu_cur,kappa_cur,omega_cur,m_cur,alpha_cur,N)
        samplesU.append(X.T)
            
        # Count generated samples
        N_tot = N_tot + N
                
        # Evaluation of the limit state function
        geval = g(X.T)
                
        # Calculation of the likelihood ratio
        W_log = likelihood_ratio_log(X,mu_cur,kappa_cur,omega_cur,m_cur,alpha_cur)

        # Check convergence
        if gamma_hat[j] == 0:
            k_fin = len(alpha_cur)     
            break

        # obtaining estimator gamma
        gamma_hat[j+1] = np.maximum(0, np.percentile(geval, rho*100))
        print(gamma_hat[j+1])

        # Indicator function
        I = (geval<=gamma_hat[j+1])

        # EM algorithm
        [mu, kappa, m, omega, alpha] = EMvMFNM(X[I,:].T, np.exp(W_log[I,:]), k_init)

        # remove unnecessary components
        if min(alpha)<=0.01:
            ind   = np.where(alpha>0.01)
            mu    = mu[:,ind]
            kappa = kappa[ind]
            m     = m[ind]
            omega = omega[ind]
            alpha = alpha[ind]

        # Assigning updated parameters            
        mu_hat    = mu.T
        kappa_hat = kappa
        m_hat     = m
        omega_hat = omega
        alpha_hat = alpha
        
    # store the needed steps
    l = j

    # %% Calculation of Probability of failure
    I  = (geval<=gamma_hat[j])
    Pr = 1/N*np.sum(np.exp(W_log[I,:])) 

    # %% transform the samples to the physical/original space
    samplesX = list()
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        if distr.Marginals[0].Name.lower() == 'standardnormal':
            for i in range(l):
                samplesX.append( samplesU[i][:,:] )
        
        else:
            for i in range(l):
                samplesX.append( distr.U2X(samplesU[i][:,:]) )

    else:
        if distr[0].Name.lower() == 'standardnormal':
            for i in range(l):
                samplesX.append( samplesU[i][:,:] )
        else:
            for i in range(l):
                samplesX.append( u2x(samplesU[i][:,:]) )

    return [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin]

# --------------------------------------------------------------------------
# Returns uniformly distributed samples from the surface of an
# n-dimensional hypersphere 
# --------------------------------------------------------------------------
# N: # samples
# n: # dimensions
# R: radius of hypersphere
# --------------------------------------------------------------------------
def hs_sample(N,n,R):

    Y = sp.stats.norm.rvs(size=(n,N)) # randn(n,N)
    Y = Y.T

    norm = np.tile(np.sqrt(np.sum(Y**2,axis=1)),[1,n])

    # X = np.matmul(Y/norm,R)
    X = Y/norm*R
    return X

# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher-Nakagami mixture
# --------------------------------------------------------------------------
def vMFNM_sample(mu,kappa,omega,m,alpha,N):

    [k, dim] = np.shape(mu)

    if k==1:
        # sampling the radius
        #     pd=makedist('Nakagami','mu',m,'omega',omega)
        #     R=pd.random(N,1)
        R = np.sqrt(sp.stats.gamma.rvs(a=m,scale=omega/m,size=[N,1]))
        
        # sampling on unit hypersphere
        X_norm = vsamp(mu.T,kappa,N)
        
    else:
        # Determine number of samples from each distribution
        z = np.sum(dummyvar(np.random.choice(range(k),N,True,alpha)),axis=0)
        k = len(z)
        
        # Generation of samples
        R = np.zeros([N,1])
        R_last = 0
        X_norm = np.zeros([N,dim])
        X_last = 0
        
        for p in range(k):
            # sampling the radius
            R[R_last:R_last+z[p],:] = np.sqrt(sp.stats.gamma.rvs(a=m[:,p],scale=omega[:,p]/m[:,p],size=[z[p],1]))
            R_last = R_last + z[p]
            
            # sampling on unit hypersphere
            X_norm[X_last:X_last+z[p],:] = vsamp(mu[p,:].T,kappa[p],z[p])
            X_last = X_last+z[p]
            
            # clear pd

    # Assign sample vector
    X = R*X_norm # bsxfun(@times,R,X_norm)
    return X

# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher distribution
# --------------------------------------------------------------------------
def vsamp(center, kappa, n):

    # d > 1 of course
    d  = np.size(center,axis=0)			# Dimensionality
    l  = kappa				# shorthand
    t1 = np.sqrt(4*l*l + (d-1)*(d-1))
    b  = (-2*l + t1 )/(d-1)

    x0 = (1-b)/(1+b)

    X = np.zeros([n,d])

    m = (d-1)/2
    c = l*x0 + (d-1)*np.log(1-x0*x0)

    for i in range(n):
        t = -1000 
        u = 1

        while t < np.log(u):
            z = sp.stats.beta.rvs(m,m)			# z is a beta rand var
            u = sp.stats.uniform.rvs()	# u is unif rand var
            w = (1 - (1+b)*z)/(1 - (1-b)*z)
            t = l*w + (d-1)*np.log(1-x0*w) - c
        
        v         = hs_sample(1,d-1,1)                                      
        # X[i,:d-1] = np.matmul(np.sqrt(1-w*w),v.T)
        X[i,:d-1] = np.sqrt(1-w*w)*v
        X[i,d-1]    = w

    [v,b] = house(center)
    Q = np.eye(d) - b*np.matmul(v,v.T)

    for i in range(n):
        tmpv = np.matmul(Q,X[i,:].T)
        X[i,:] = tmpv.T

    return X

# --------------------------------------------------------------------------
# X,mu,kappa
# Returns the von Mises-Fisher mixture log pdf on the unit hypersphere
# --------------------------------------------------------------------------
def vMF_logpdf(X,mu,kappa):

    d = np.size(X, axis=0)
    n = np.size(X, axis=1)

    if kappa == 0:
        A = np.log(d) + np.log(np.pi**(d/2)) - sp.special.gammaln(d/2+1)
        y = -A*np.ones([1,n])
    elif kappa > 0:
        c = (d/2-1)*np.log(kappa)-(d/2)*np.log(2*np.pi)-logbesseli(d/2-1,kappa)
        q = np.matmul((mu*kappa).T,X) # bsxfun(@times,mu,kappa)'*X
        y = q + c.T                   # bsxfun(@plus,q,c')
    else:
        raise ValueError('Kappa must not be negative!')

    return y

# --------------------------------------------------------------------------
# Returns the value of the log-nakagami-pdf
# --------------------------------------------------------------------------
def nakagami_logpdf(X,m,om):

    y = np.log(2) + m*(np.log(m)-np.log(om)-X**2/om) + np.log(X)*(2*m-1) - sp.special.gammaln(m)
    return y

# --------------------------------------------------------------------------
# likelihood_ratio_log()
# --------------------------------------------------------------------------
def likelihood_ratio_log(X,mu,kappa,omega,m,alpha):
    k       = len(alpha)
    [N,dim] = np.shape(X)
    R       = np.sqrt(np.sum(X*X,axis=1)).reshape(-1,1)

    if k==1:
        
        # log pdf of vMF distribution
        logpdf_vMF = vMF_logpdf((X/R).T,mu.T,kappa).T
        # log pdf of Nakagami distribution
        logpdf_N   = nakagami_logpdf(R,m,omega)
        # log pdf of weighted combined distribution
        h_log      = logpdf_vMF + logpdf_N
        
    else:
        
        logpdf_vMF = np.zeros([N,k])
        logpdf_N   = np.zeros([N,k])
        h_log      = np.zeros([N,k])
        
        # log pdf of distributions in the mixture 
        for p in range(k):
            # log pdf of vMF distribution
            logpdf_vMF[:,p] = vMF_logpdf((X/R).T, mu[p,:].T,kappa[p]).squeeze()
            # log pdf of Nakagami distribution
            logpdf_N[:,p]   = nakagami_logpdf(R,m[:,p],omega[:,p]).squeeze()
            # log pdf of weighted combined distribution
            h_log[:,p]      = logpdf_vMF[:,p] + logpdf_N[:,p] + np.log(alpha[p])
        
        # mixture log pdf
        h_log = logsumexp(h_log,1)

    # unit hypersphere uniform log pdf
    A   = np.log(dim) + np.log(np.pi**(dim/2)) - sp.special.gammaln(dim/2+1)
    f_u = -A

    # chi log pdf
    f_chi=np.log(2)*(1-dim/2) + np.log(R)*(dim-1) - 0.5*R**2 - sp.special.gammaln(dim/2)

    # logpdf of the standard distribution (uniform combined with chi
    # distribution)
    f_log = f_u + f_chi

    W_log = f_log - h_log

    return W_log

# --------------------------------------------------------------------------
# HOUSE Returns the householder transf to reduce x to b*e_n 
#
# [V,B] = HOUSE(X)  Returns vector v and multiplier b so that
# H = eye(n)-b*v*v' is the householder matrix that will transform
# Hx ==> [0 0 0 ... ||x||], where  is a constant.
# --------------------------------------------------------------------------
def house(x):
    x = x.squeeze()
    n = len(x)

    s = np.matmul(x[:n-1].T,x[:n-1])
    v = np.concatenate([x[:n-1], np.array([1.0])]).squeeze()

    if s == 0:
        b = 0
    else:
        m = np.sqrt(x[n-1]*x[n-1] + s)
    
        if x[n-1] <= 0:
            v[n-1] = x[n-1]-m
        else:
            v[n-1] = -s/(x[n-1]+m)

        b = 2*v[n-1]*v[n-1]/(s + v[n-1]*v[n-1])
        v = v/v[n-1]
    
    v = v.reshape(-1,1)
    return [v,b]

# --------------------------------------------------------------------------
# log of the Bessel function, extended for large nu and x
# approximation from Eqn 9.7.7 of Abramowitz and Stegun
# http://www.math.sfu.ca/~cbm/aands/page_378.htm
# --------------------------------------------------------------------------
def logbesseli(nu,x):

    if nu == 0: # special case when nu=0
        logb = np.log(sp.special.iv(nu,x)) # besseli
    else: # normal case
        # n    = np.size(x, axis=0)
        n      = 1      # since x is always scalar here
        frac   = x/nu
        
        square = np.ones(n) + frac**2
        root   = np.sqrt(square)
        eta    = root + np.log(frac) - np.log(np.ones(n)+root)
        logb   = - np.log(np.sqrt(2*np.pi*nu)) + nu*eta - 0.25*np.log(square)

    return [logb]

# --------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
#   By default dim = 0 (columns).
# Written by Michael Chen (sth4nth@gmail.com).
# --------------------------------------------------------------------------
def logsumexp(x, dim=0):
    # subtract the largest in each column
    y = np.max(x, axis=dim).reshape(-1,1)
    x = x - y
    s = y + np.log(np.sum(np.exp(x),axis=dim)).reshape(-1,1)
    # if a bug occurs here, maybe find a better translation from matlab:
    i = np.where(np.invert(np.isfinite(y).squeeze()))
    s[i] = y[i] 
    
    return s

# --------------------------------------------------------------------------
# Translation of the Matlab-function "dummyvar()" to Python
# --------------------------------------------------------------------------
def dummyvar(idx):
    n = np.max(idx)+1
    d = np.zeros([len(idx),n],int)
    for i in range(len(idx)):
        d[i,idx[i]] = 1
    return d
# %% END FILE