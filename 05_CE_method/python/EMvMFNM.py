import numpy as np
import scipy as sp
from scipy import cluster
"""
---------------------------------------------------------------------------
Perform soft EM algorithm for fitting the von Mises-Fisher-Nakagami mixture model.
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
* X   : data matrix (dimensions x Number of samples)
* W   : vector of likelihood ratios for weighted samples
* nGM : number of vMFN-distributions in the mixture
---------------------------------------------------------------------------
Output:
* mu    :
* kappa :
* m     :
* omega :
* alpha :
---------------------------------------------------------------------------
Based on:
1. "EM Demystified: An Expectation-Maximization Tutorial"
   Yihua Chen and Maya R. Gupta
   University of Washington, Dep. of EE (Feb. 2010)
---------------------------------------------------------------------------
"""
def EMvMFNM(X,W,k):
    # reshaping just to be sure
    W = W.reshape(-1,1)

    ## initialization
    M = initialization(X, k)

    R      = np.sqrt(np.sum(X*X)).T  # R=sqrt(sum(X.^2))'
    X_norm = X/R.T                   # X_norm=(bsxfun(@times,X,1./R'))

    tol       = 1e-5
    maxiter   = 500
    llh       = np.full([2,maxiter],-np.inf)
    converged = False
    t         = 0

    ## soft EM algorithm
    while (not converged) and (t+1 < maxiter):
        t = t+1

        # [~,label(:)] = max(M,[],2)
        # u = unique(label)   # non-empty components
        # if size(M,2) ~= size(u,2)
        #     M = M(:,u)   # remove empty components
        if t > 1:
            con1      = abs(llh[0,t-1]-llh[0,t-2]) < tol*abs(llh[0,t-1])
            con2      = abs(llh[1,t-1]-llh[1,t-2]) < tol*100*abs(llh[1,t-1])
            converged = min(con1, con2)

        [mu, kappa, m, omega, alpha]  = maximization(X_norm, W, R, M)
        [M, llh[:,t]] = expectation(X_norm, W, R, mu, kappa, m, omega, alpha)

    if converged:
        print('Converged in', t,'steps.')
    else:
        print('Not converged in ', maxiter, ' steps.')

    return [mu,kappa,m,omega,alpha]
# %% END EMvMFNM ----------------------------------------------------------------


# --------------------------------------------------------------------------
# Initialization with k-means algorithm 
# --------------------------------------------------------------------------
def initialization(X, k):

    [_,idx] = sp.cluster.vq.kmeans2(X.T, k, iter=10)
    M   = dummyvar(idx)
    return M

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def expectation(X,W,R,mu,kappa,m,omega,alpha):
    n = np.size(X, axis=1)
    k = np.size(mu, axis=1)

    logvMF      = np.zeros([n,k])
    lognakagami = np.zeros([n,k])
    logpdf      = np.zeros([n,k])

    # logpdf
    for i in range(k):
        logvMF[:,i]      = logvMFpdf(X,mu[:,i],kappa[i]).T
        lognakagami[:,i] = lognakagamipdf(R,m[i],omega[i])
        logpdf[:,i]      = logvMF[:,i] + lognakagami[:,i] + np.log(alpha[i])

    # Matrix of posterior probabilities
    T    = logsumexp(logpdf,2)
    logM = logpdf - T             # logM = bsxfun(@minus,logpdf,T)
    M    = np.exp(logM)
    M(M<1e-3) = 0                 # TODO: does it work? I doubt it ;-)
    M    = M/np.sum(M, axis=1)    # M=bsxfun(@times,M,1./sum(M,2))

    # loglikelihood as tolerance criterion
    logvMF_weighted      = logvMF + np.log(alpha) # bsxfun(@plus,logvMF,log(alpha))
    lognakagami_weighted = lognakagami + np.log(alpha) # bsxfun(@plus,lognakagami,log(alpha))
    T_vMF                = logsumexp(logvMF_weighted,2)
    T_nakagami           = logsumexp(lognakagami_weighted,2)
    llh1 = np.array([sum(W*T_vMF)/sum(W), sum(W*T_nakagami)/sum(W)])
    llh  = llh1
return [M,llh]

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def maximization(X,W,R,M):
    M  = W*M                   # repmat(W,1,size(M,2)).*M
    d  = np.size(X, axis=0)
    nk = np.sum(M, axis=0)

    #  distribution weights
    alpha = nk/sum(W)
    
    # mean directions
    mu_unnormed = X*M
    norm_mu     = np.sqrt(np.sum(mu_unnormed*mu_unnormed, axis=0))
    mu          = mu_unnormed/norm_mu # bsxfun(@times,mu_unnormed,1./norm_mu)

    # approximated concentration parameter 
    xi    = np.minimum(norm_mu/nk,0.95)
    kappa = (xi*d-xi**3)/(1-xi**2)

    # spread parameter 
    omega = np.matmul(M.T,R*R).T/np.sum(M)

    # approximated shape parameter
    mu4       = np.matmul(M.T, R*R*R*R).T/np.sum(M)
    m         = omega**2/(mu4-omega**2)
    m(m<0)    = d/2
    m(m>20*d) = d/2

    return [mu, kappa, m, omega, alpha]

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def logvMFpdf(X, mu, kappa):
    d = np.size(X, axis=0)
    if kappa == 0: 
        # unit hypersphere uniform log pdf
        A = np.log(d)+np.log(np.pi^(d/2))-gammaln(d/2+1)
        y = -A
    elif kappa > 0:
        c = (d/2-1)*np.log(kappa) - (d/2)*np.log(2*np.pi) - logbesseli(d/2-1,kappa)
        q = (mu*kappa).T*X # bsxfun(@times,mu,kappa)'*X
        y = q + c.T # bsxfun(@plus,q,c')
    else:
        raise ValueError('Concentration parameter kappa must not be negative!')

    return y

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def lognakagamipdf(X,m,om):

    y = np.log(2) + m*(np.log(m)-np.log(om)-X*X/om) + np.log(X)*(2*m-1) - gammaln(m)
    return y

# --------------------------------------------------------------------------
# log of the Bessel function, extended for large nu and x
# approximation from Eqn 9.7.7 of Abramowitz and Stegun
# http://www.math.sfu.ca/~cbm/aands/page_378.htm
# --------------------------------------------------------------------------
def logbesseli(nu,x):

    if nu == 0: # special case when nu=0
        logb = np.log(besseli(nu,x))
    else: # normal case
        n    = np.size(x, axis=0)
        frac = x/nu
        
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