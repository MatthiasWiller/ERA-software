import numpy as np
import scipy.stats
"""
---------------------------------------------------------------------------
Perform soft EM algorithm for fitting the Gaussian mixture model
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
---------------------------------------------------------------------------
Input:
* X       :
* W       :
* nGM     :
---------------------------------------------------------------------------
Output:
* mu    : 
* Sigma :
* pi    :
---------------------------------------------------------------------------
"""
def EMGM(X, W, nGM):
    ## initialization
    R = initialization(X, nGM)

    tol       = 1e-5
    maxiter   = 500
    llh       = -np.inf(maxiter)
    converged = false
    t         = 1

    ## soft EM algorithm
    while (not converged) and t < maxiter:
        t = t+1   
        
        # [~,label(:)] = max(R,[],2)
        # u = unique(label)   # non-empty components
        # if size(R,2) ~= size(u,2):
        #     R = R[:,u]   # remove empty components
        if t > 1:
            converged = abs(llh(t)-llh(t-1)) < tol*abs(llh(t))
        
        [mu, si, pi] = maximization(X,W,R)
        [R, llh(t)] = expectation(X, W, mu, si, pi)
    

    if converged:
        print('Converged in', t-1,'steps.')
    else:
        print('Not converged in ', maxiter, ' steps.')

    return [mu, si, pi]
## END EMGM ----------------------------------------------------------------

# --------------------------------------------------------------------------
# Initialization with k-means algorithm 
# --------------------------------------------------------------------------
def initialization(X, nGM):

    idx = scipy.cluster.vq.kmeans2(X.T, nGM) # idx = kmeans(X.T,nGM,'Replicates',10)
    R   = dummyvar(idx)
    return R

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def expectation(X, W, mu, si, pi):
    n = np.size(X, axis=1)
    k = np.size(mu, axis=1)

    logpdf = np.zeros((n,k))
    for i in range(k):
        logpdf[:,i] = loggausspdf(X, mu[:,i], si[:,:,i])


    logpdf = logpdf + np.log(np.pi)     # logpdf = bsxfun(@plus,logpdf,log(pi))
    T      = logsumexp(logpdf,1)
    llh    = np.sum(W*T)/np.sum(W)
    logR   = logpdf - T              # logR = bsxfun(@minus,logpdf,T)
    R      = np.exp(logR)
    return [R, llh]


# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def maximization(X, W, R):
    R = W*R
    d = np.size(X, axis=0)
    k = np.size(R, axis=1)

    nk = sum(R,axis=0)
    w  = nk/sum(W)
    mu = X*R/nk              # mu = bsxfun(@times, X*R, 1./nk)

    Sigma = np.zeros((d,d,k))
    sqrtR = np.sqrt(R)
    for i in range(k):
        Xo = X-mu[:,i]        # Xo = bsxfun(@minus,X,mu(:,i))
        Xo = Xo*sqrtR[:,i]    # Xo = bsxfun(@times,Xo,sqrtR(:,i)')
        Sigma[:,:,i] = Xo*Xo/nk[i]
        Sigma[:,:,i] = Sigma[:,:,i]+np.eye(d)*(1e-6) # add a prior for numerical stability

    return [mu, Sigma, w]

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def loggausspdf(X, mu, Sigma):
    d = np.size(X, axis=0)
    X = X-mu                   # X = bsxfun(@minus,X,mu)
    [U,~]= np.chol(Sigma)
    Q = U'\X #TODO: solve this here
    q = dot(Q,Q,1)  # quadratic term (M distance)
    c = d*np.log(2*np.pi)+2*np.sum(np.log(np.diag(U)))   # normalization constant
    y = -(c+q)/2

    return y

# --------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
#   By default dim = 0 (columns).
# Written by Michael Chen (sth4nth@gmail.com).
# --------------------------------------------------------------------------
def logsumexp(x, dim=0):
    # subtract the largest in each column
    y = np.max(x,[],dim)
    x = x - y                  #x = bsxfun(@minus,x,y)
    s = y + np.log(np.sum(np.exp(x),dim))
    i = find(not(isfinite(y)))
    if not(isempty[i]):
        s[i] = y[i]
    
    return s
## END FILE