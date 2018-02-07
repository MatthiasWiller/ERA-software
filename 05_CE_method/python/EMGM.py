import numpy as np
import scipy.stats
"""
---------------------------------------------------------------------------
Basic algorithm
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
* algtype :
---------------------------------------------------------------------------
Output:
* mu : 
* 
---------------------------------------------------------------------------
"""
def EMGM(X, W, nGM, algtype):
    ## initialization
    R = initialization(X,init)

    tol = 1e-5
    maxiter = 500
    llh = -inf(1,maxiter)
    converged = false
    t = 1

    ## soft EM algorithm
    while (not converged) and t < maxiter:
        t = t+1   
        
        [~,label(:)] = max(R,[],2)
        u = unique(label)   # non-empty components
        if size(R,2) ~= size(u,2):
            R = R(:,u)   # remove empty components
        else:
            converged = abs(llh(t)-llh(t-1)) < tol*abs(llh(t))
        
        model = maximization(X,W,R)
        [R, llh(t)] = expectation(X,W,model)
    

    if converged:
        print('Converged in', t-1,'steps.')
    else:
        print('Not converged in ', maxiter, ' steps.')


def initialization(X,init):

    [~,n] = size(X)

    if strcmp('DBSCAN',init.type): # Initialization with cluster centers
        
        m=init.centers'
        k=size(m,2)
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1)
        R = full(sparse(1:n,label,1,n,k,n))
        
    elif strcmp('RAND',init.type): # Random initialization 
        
        idx = randsample(n,init.nGM)
        m = X(:,idx)
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1)
        [u,~,label] = unique(label)
        while init.nGM ~= length(u)
            idx = randsample(n,init.nGM)
            m = X(:,idx)
            [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1)
            [u,~,label] = unique(label)
        end
        
        R = full(sparse(1:n,label,1,n,init.nGM,n))
        
    elif strcmp('KMEANS',init.type): # Initialization with k-means algorithm 
        
        idx=kmeans(X',init.nGM,'Replicates',10)
        
        R=dummyvar(idx)

    return R


def expectation(X, W, model):
    mu = model.mu
    si = model.si
    pi = model.pi

    n = size(X,2)
    k = size(mu,2)

    logpdf = zeros(n,k)
    for i = 1:k
        logpdf(:,i) = loggausspdf(X,mu(:,i),si(:,:,i))
    end

    logpdf = bsxfun(@plus,logpdf,log(pi))
    T = logsumexp(logpdf,2)
    llh = sum(W.*T)/sum(W)
    logR = bsxfun(@minus,logpdf,T)
    R = exp(logR)
    return [R, llh]

def maximization(X, W, R):
    R = repmat(W,1,size(R,2)).*R
    [d,~] = size(X)
    k = size(R,2)

    nk = sum(R,1)
    w = nk/sum(W)
    mu = bsxfun(@times, X*R, 1./nk)

    Sigma = zeros(d,d,k)
    sqrtR = sqrt(R)
    for i = 1:k
        Xo = bsxfun(@minus,X,mu(:,i))
        Xo = bsxfun(@times,Xo,sqrtR(:,i)')
        Sigma(:,:,i) = Xo*Xo'/nk(i)
        Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-6) # add a prior for numerical stability
    end

    model.mu = mu
    model.si = Sigma
    model.pi = w

    return model

def loggausspdf(X, mu, Sigma):
    d = size(X,1)
    X = bsxfun(@minus,X,mu)
    [U,~]= chol(Sigma)
    Q = U'\X
    q = dot(Q,Q,1)  # quadratic term (M distance)
    c = d*log(2*pi)+2*sum(log(diag(U)))   # normalization constant
    y = -(c+q)/2

    return y

def logsumexp(x, dim):
    # Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
    #   By default dim = 1 (columns).
    # Written by Michael Chen (sth4nth@gmail.com).
    if nargin == 1
        # Determine which dimension sum will use
        dim = find(size(x)~=1,1)
        if isempty(dim), dim = 1 end
    end

    # subtract the largest in each column
    y = max(x,[],dim)
    x = bsxfun(@minus,x,y)
    s = y + log(sum(exp(x),dim))
    i = find(~isfinite(y))
    if ~isempty(i)
        s(i) = y(i)
    end

    return s
## END