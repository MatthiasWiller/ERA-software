function [model,init] = EMvMFNM(X,W,init)
% Perform soft EM algorithm for fitting the von Mises-Fisher-Nakagami mixture model.
%   X: d x N data matrix (dimensions x Number of samples)
%   W: N x 1 vector of likelihood ratios for weighted samples
%   init: struct or vector to define initialization of EM algorithm

%% initialization
M = initialization(X,init);

R=sqrt(sum(X.^2))';
X_norm=(bsxfun(@times,X,1./R'));

tol = 1e-3;
maxiter = 500;
llh = -inf(2,maxiter);
converged = false;
t = 1;

%% soft EM algorithm

while ~converged && t < maxiter
    t = t+1;
    [~,label(:)] = max(M,[],2);
    u = unique(label);   % non-empty components
    if size(M,2) ~= size(u,2)
        M = M(:,u);   % remove empty components
    elseif t>2
        con1 = abs(llh(1,t-1)-llh(1,t-2)) < tol*abs(llh(1,t-1));
        con2=abs(llh(2,t-1)-llh(2,t-2))<tol*100*abs(llh(2,t-1));
        converged=min(con1,con2);
    end
    model = maximization(X_norm,W,R,M);
    [M,llh(:,t)] = expectation(X_norm,W,model,R);
end
if converged
    fprintf(' EM converged in %d steps\n',t-1);
else
    fprintf(' EM not converged in %d steps\n',maxiter);
end

function M = initialization(X,init)
[d,n] = size(X);

if strcmp('DBSCAN',init.type) % Initialization with cluster centers
    
    m=init.centers';
    k=size(m,2);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    M = full(sparse(1:n,label,1,n,k,n));
    
elseif strcmp('RAND',init.type) % Random initialization
    
    label = ceil(init.k*rand(1,n));
    [u,~,label] = unique(label);
    while init.k ~= length(u)
        label = ceil(init.k*rand(1,n));
        [u,~,label] = unique(label);
    end
    
    M = full(sparse(1:n,label,1,n,init.k,n));
    
%     idx = randsample(n,init.k);
%     m = X(:,idx);
%     [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
%     [u,~,label] = unique(label);
%     while init.k ~= length(u)
%         idx = randsample(n,init.k);
%         m = X(:,idx);
%         [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
%         [u,~,label] = unique(label);
%     end
%     
%     M = full(sparse(1:n,label,1,n,init.k,n));
    
elseif strcmp('KMEANS',init.type) % Initialization with k-means algorithm
    
    idx = kmeans(X',init.k,'MaxIter',10000,'Replicates',10,'Distance','cosine');
    M=dummyvar(idx);
    
end

function [M,llh] = expectation(X,W,model,R)
mu = model.mu;
kappa = model.kappa;
m=model.m;
omega=model.omega;
alpha=model.alpha;

n = size(X,2);
k = size(mu,2);
logvMF = zeros(n,k);
lognakagami=zeros(n,k);
logpdf=zeros(n,k);

% logpdf
for i = 1:k
    logvMF(:,i) = (logvMFpdf(X,mu(:,i),kappa(i)))';
    lognakagami(:,i)=lognakagamipdf(R,m(i),omega(i));
    logpdf(:,i)=logvMF(:,i)+lognakagami(:,i)+log(alpha(i));
end

% Matrix of posterior probabilities
T = logsumexp(logpdf,2);
logM = bsxfun(@minus,logpdf,T);
M=exp(logM);
M(M<1e-3)=0;
M=bsxfun(@times,M,1./sum(M,2));

% loglikelihood as tolerance criterion
logvMF_weighted=bsxfun(@plus,logvMF,log(alpha));
lognakagami_weighted=bsxfun(@plus,lognakagami,log(alpha));
T_vMF=logsumexp(logvMF_weighted,2);
T_nakagami=logsumexp(lognakagami_weighted,2);
llh1 = [sum(W.*T_vMF)/sum(W);sum(W.*T_nakagami)/sum(W)];
llh=llh1;

function model = maximization(X,W,R,M)
M = repmat(W,1,size(M,2)).*M;
[d,~] = size(X);
nk=sum(M,1);

%  distribution weights
alpha=nk/sum(W);

% mean directions
mu_unnormed=X*M;
norm_mu=sqrt(sum(mu_unnormed.^2,1));
mu=bsxfun(@times,mu_unnormed,1./norm_mu);

% approximated concentration parameter
xi=min(norm_mu./nk,0.95);
kappa=(xi.*d-xi.^3)./(1-xi.^2);

% spread parameter
omega=(M'*R.^2)'./sum(M);

% approximated shape parameter
mu4=(M'*R.^4)'./sum(M);
m=omega.^2./(mu4-omega.^2);
m(m<0)=d/2;
m(m>20*d)=d/2;

% assigning model
model.mu = mu;
model.kappa= kappa;
model.m=m;
model.omega=omega;
model.alpha=alpha;

function y = logvMFpdf(X, mu, kappa)

d = size(X,1);
if kappa==0
    % unit hypersphere uniform log pdf
    A=log(d)+log(pi^(d/2))-gammaln(d/2+1);
    y=-A;
elseif kappa>0
    c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
    q = bsxfun(@times,mu,kappa)'*X;
    y = bsxfun(@plus,q,c');
else
    error('concentration parameter must not be negative')
end

function y = lognakagamipdf(X,m,om)

y=log(2)+m*(log(m)-log(om)-X.^2./om)+log(X).*(2*m-1)-gammaln(m);

function [logb] = logbesseli(nu,x)
% log of the Bessel function, extended for large nu and x
% approximation from Eqn 9.7.7 of Abramowitz and Stegun
% http://www.math.sfu.ca/~cbm/aands/page_378.htm

if nu==0 % special case when nu=0
    logb=log(besseli(nu,x));
else % normal case
    n=size(x,1);
    frac = x./nu;
    
    square = ones(n,1) + frac.^2;
    root = sqrt(square);
    eta = root + log(frac) - log(ones(n,1)+root);
    logb = - log(sqrt(2*pi*nu)) + nu.*eta - 0.25*log(square);
end

function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end
