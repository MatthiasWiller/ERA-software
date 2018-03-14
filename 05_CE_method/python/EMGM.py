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

    idx = kmeans(X.T,nGM,'Replicates',10)
    R   = dummyvar(idx)
    return R

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def expectation(X, W, mu, si, pi):

    n = np.size(X,2)
    k = np.size(mu,2)

    logpdf = np.zeros(n,k)
    for i in range(k):
        logpdf[:,i] = loggausspdf(X,mu[:,i],si[:,:,i])


    logpdf = bsxfun(@plus,logpdf,log(pi))
    T = logsumexp(logpdf,2)
    llh = sum(W.*T)/sum(W)
    logR = bsxfun(@minus,logpdf,T)
    R = exp(logR)
    return [R, llh]


# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def maximization(X, W, R):
    R = repmat(W,1,np.size(R,axis=1)).*R
    d = np.size(X, axis=0)
    k = np.size(R, axis=1)

    nk = sum(R,axis=0)
    w  = nk/sum(W)
    mu = bsxfun(@times, X*R, 1./nk)

    Sigma = np.zeros((d,d,k))
    sqrtR = np.sqrt(R)
    for i in range(k):
        Xo = bsxfun(@minus,X,mu(:,i))
        Xo = bsxfun(@times,Xo,sqrtR(:,i)')
        Sigma(:,:,i) = Xo*Xo'/nk(i)
        Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-6) # add a prior for numerical stability

    return [mu, Sigma, w]

# --------------------------------------------------------------------------
# ...
# --------------------------------------------------------------------------
def loggausspdf(X, mu, Sigma):
    d = np.size(X, axis=0)
    X = bsxfun(@minus,X,mu)
    [U,~]= chol(Sigma)
    Q = U'\X
    q = dot(Q,Q,1)  # quadratic term (M distance)
    c = d*log(2*pi)+2*sum(log(diag(U)))   # normalization constant
    y = -(c+q)/2

    return y

# --------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
#   By default dim = 1 (columns).
# Written by Michael Chen (sth4nth@gmail.com).
# --------------------------------------------------------------------------
def logsumexp(x, dim):
    
    if nargin == 1:
        # Determine which dimension sum will use
        dim = find(size(x)~=1,1)
        if isempty(dim):
            dim = 1

    # subtract the largest in each column
    y = max(x,[],dim)
    x = bsxfun(@minus,x,y)
    s = y + log(sum(exp(x),dim))
    i = find(~isfinite(y))
    if ~isempty(i):
        s[i] = y[i]
    
    return s
## END FILE