function result = CEIS_new_method(input)

% tolerance criterion
tolCOV = 1.5;

% shorthand
max_iter=input.max_iter;
dim=input.dim;
N=input.N;
cykles=input.cykles;
f=input.f;
version=input.version;

% Vectors for simulation results
Pr=zeros(cykles,1);
l=zeros(cykles,1);
N_tot=zeros(cykles,1);
k_fin=zeros(cykles,1);

% initial nakagami parameters (make it equal to chi distribution)
omega_init=dim; % spread parameter
m_init=dim/2; % shape parameter

% initial von Mises-Fisher parameters
kappa_init=0; % Concentration parameter (zero for uniform distribution)
mu_init=hs_sample(1, dim,1); % Initial mean sampled on unit hypersphere

% initial disribution weight
alpha_init=1;

gamma_hat = zeros(max_iter+1,cykles);


for i=1:cykles
    
    %% Initializing parameters
    mu_hat=mu_init;
    kappa_hat=kappa_init;
    omega_hat=omega_init;
    m_hat=m_init;
    gamma_hat(1,i) = 1;
    alpha_hat=alpha_init;
    sigmak=zeros(max_iter,1);
    
    %% Iteration
    
    for j=1:max_iter
        
        % save parameters from previous step
        mu_cur=mu_hat;
        kappa_cur=kappa_hat;
        omega_cur=omega_hat;
        m_cur=m_hat;
        alpha_cur=alpha_hat;
        
        % Generate samples
        X=vMFNM_sample(mu_cur,kappa_cur,omega_cur,m_cur,alpha_cur,N);
        
%         figure('Position', [100, 100,750,750]);
%         
%         if j==1
%             plot(mu_hat(1,:),'color','k')
%             ylim([-0.4 0.4])
%             hold on
%             refline([0 1/sqrt(dim)])
%             refline([0 -1/sqrt(dim)])
%             box on
%             set(gca, 'FontSize', 16)
%         else
%             plot(mu_hat(1,:),'color','k')
%             hold on
%             plot(mu_hat(2,:),'color','k')
%             ylim([-0.4 0.4])
%             refline([0 1/sqrt(dim)])
%             refline([0 -1/sqrt(dim)])
%             box on
%             set(gca, 'FontSize', 16)
%         end
        
        %         plot(X(:,1),X(:,2),'x')
        %         hold on
        
        % Count generated samples
        N_tot(i)=N_tot(i)+N;
        
        % Evaluation of the limit state function
        g=f(X);
        
        % Check convergence
        I=g<=0;
        if nnz(I)>=N/10
            k_fin(i)=length(dist.kappa);
            Y=I.*exp(vMFNM_logpdf(X)-vMFNM_logpdf(X,mu_cur,kappa_cur,omega_cur,m_cur,alpha_cur));
            break;
        end
        
        % compute sigma and weights
        if j == 1
            mu_g=mean(g);
            sigmak(1) = 50*mu_g;
            sigma2 = fminbnd(@(x)abs(std(normcdf(-g/x))/mean(normcdf(-g/x))-tolCOV),0,sigmak(j));
            sigmak(j+1) = sigma2;
            W = normcdf(-g/sigmak(j+1));
        else
            C=exp(vMFNM_logpdf(X)-vMFNM_logpdf(X,mu_cur,kappa_cur,omega_cur,m_cur,alpha_cur));
            fmin=@(x)abs(std(normcdf(-g/x).*C)/mean(normcdf(-g/x).*C)-tolCOV);
            sigma2 = fminbnd(fmin,0,sigmak(j));
            sigmak(j+1) = sigma2;
            W = normcdf(-g/sigmak(j+1)).*C;
        end
        
        % Normalize weights
        W=W./sum(W);
        
        % EM initialization with kmeans algorithm
        init.type='KMEANS';
        init.k=4;
        
        % EM algorithm
        dist=EMvMFNM(X',W,init);
        
        % remove unnecessary components
        if min(dist.alpha)<=0.01
            ind=find(dist.alpha>0.01);
            dist.mu=dist.mu(:,ind);
            dist.kappa=dist.kappa(ind);
            dist.m=dist.m(ind);
            dist.omega=dist.omega(ind);
            dist.alpha=dist.alpha(ind);
        end
        
        % Assigning updated parameters
        mu_hat=dist.mu';
        kappa_hat=dist.kappa;
        m_hat=dist.m;
        omega_hat=dist.omega;
        alpha_hat=dist.alpha;
        
    end
    
    % store the needed steps
    l(i)=j;
    
    %% Calculation of Probability of failure
    Pr(i)=1/N*sum(Y);
    
    %% Output of progress
    if mod(i,cykles/100)==0
        sprintf('%d%% completed',i/cykles*100)
    end
    
end

% plot final weighted samples and limit state function
if cykles ==1 && dim<3
Pr
figure('Position', [100, 100,750,750]);
hold on
%gp=@(x,y)min([0.1.*(x-y).^2-(x+y)./sqrt(2)+3,0.1.*(x-y).^2+(x+y)./sqrt(2)+3,x-y+7./sqrt(2),y-x+7./sqrt(2)],[],2);
gp=@(x,y)min(input.beta-1/sqrt(input.dim)*(x+y),input.beta+1/sqrt(input.dim)*(x+y));
plot(X(:,1),X(:,2),'o','color','k')
box on
fcontour(gp,[-6 6],'LevelList',0,'LineColor','k','MeshDensity',1000);
set(gca, 'FontSize', 16)
end

result.Version =version;
result.Pr=Pr;
result.E_Pr=mean(Pr);
result.CoV_Pr=std(Pr)/result.E_Pr;
result.l=l;
result.E_l=mean(l);
result.N_tot=N_tot;
result.E_N_tot=mean(N_tot);
result.gamma_hat=gamma_hat;
result.k_fin=k_fin;
result.E_k_fin=mean(k_fin);

function X = hs_sample(N,n,R)
% Returns uniformly distributed samples from the surface of an
% n-dimensional hypersphere

% N: # samples
% n: # dimensions
% R: radius of hypersphere

Y = randn(n,N);
Y=Y';

norm=repmat(sqrt(sum(Y.^2,2)),[1 n]);

X=Y./norm*R;

function X=vMFNM_sample(mu,kappa,omega,m,alpha,N)
% Returns samples from the von Mises-Fisher-Nakagami mixture

[k,dim]=size(mu);

if k==1
    
    % sampling the radius
    %     pd=makedist('Nakagami','mu',m,'omega',omega);
    %     R=pd.random(N,1);
    R=sqrt(gamrnd(m,omega./m,N,1));
    
    % sampling on unit hypersphere
    X_norm=vsamp(mu',kappa,N);
    
else
    
    % Determine number of samples from each distribution
    z=sum(dummyvar(randsample(k,N,true,alpha)));
    k=length(z);
    
    % Generation of samples
    R=zeros(N,1);
    R_last=0;
    X_norm=zeros(N,dim);
    X_last=0;
    
    for p=1:k

        % sampling the radius
        R(R_last+1:R_last+z(p))=sqrt(gamrnd(m(p),omega(p)./m(p),z(p),1));
        R_last=R_last+z(p);
        
        % sampling on unit hypersphere
        X_norm(X_last+1:X_last+z(p),:)=vsamp(mu(p,:)',kappa(p),z(p));
        X_last=X_last+z(p);
        
        clear pd
    end
end

% Assign sample vector
X=bsxfun(@times,R,X_norm);

function X = vsamp(center, kappa, n)
% Returns samples from the von Mises-Fisher distribution

% d > 1 of course
d = size(center,1);			% Dimensionality
l = kappa;				% shorthand
t1 = sqrt(4*l*l + (d-1)*(d-1));
b = (-2*l + t1 )/(d-1);

x0 = (1-b)/(1+b);

X = zeros(n,d);

m = (d-1)/2;
c = l*x0 + (d-1)*log(1-x0*x0);


y=hs_sample(n,d-1,1)';

for i=1:n
    t = -1000; u = 1;
    
    while (t < log(u))
        z = betarnd(m , m);			% z is a beta rand var
        u = rand(1);				% u is unif rand var
        w = (1 - (1+b)*z)/(1 - (1-b)*z);
        t = l*w + (d-1)*log(1-x0*w) - c;
    end
    
    %v2=hs_sample(1,d-1,1);
    X(i,1:d-1) = sqrt(1-w*w)*y(:,i);
    X(i,d) = w;
end

[v,b]=house(center);
Q = eye(d)-b*v*v';

X=(Q*X')';

% for i=1:n
%   tmpv = Q*X(i,:)';
%   X(i,:) = tmpv';
%end

function f_log=vMFNM_logpdf(X,mu,kappa,omega,m,alpha)
% Returns the pdf of the von Mises-Fisher-Nakagami mixture
% Input parameters:
%   X: N x dim data matrix  (Number of samples x dimensions)
%   von Mises-Fisher parameters:
%     mu: k x dim matrix    (mean direction)
%     kappa: 1 x k vector   (concentration parameter)
%   Nakagami parameters:
%     omega: 1 x k vector   (spread parameter)
%     m: 1 x k vector of    (shape parameter)
%   alpha: 1 x k vector     (distribution weights)
%
% If only X is given as Input, it will return the standard vMFN pdf, which
% is equivalent to the multidimensional standard  normal pdf

[N,dim]=size(X);
R=sqrt(sum(X.^2,2));

if nargin==1
    
    % unit hypersphere uniform log pdf
    A=log(dim)+log(pi^(dim/2))-gammaln(dim/2+1);
    f_u=-A;
    
    % chi log pdf
    f_chi=log(2)*(1-dim/2)+log(R)*(dim-1)-0.5*R.^2-gammaln(dim/2);
    
    % logpdf of the standard vMFN distribution
    f_log=f_u+f_chi;
    
elseif nargin==6
    k=length(alpha);
    
    if k==1
        
        % log pdf of vMF distribution
        logpdf_vMF=vMF_logpdf((bsxfun(@times,X,1./R))',mu',kappa)';
        % log pdf of Nakagami distribution
        logpdf_N=nakagami_logpdf(R,m,omega);
        % log pdf of weighted combined distribution
        f_log=logpdf_vMF+logpdf_N;
        
    else
        
        logpdf_vMF=zeros(N,k);
        logpdf_N=zeros(N,k);
        f_log=zeros(N,k);
        
        % log pdf of distributions in the mixture
        for p=1:k
            % log pdf of vMF distribution
            logpdf_vMF(:,p)=vMF_logpdf((bsxfun(@times,X,1./R))',mu(p,:)',kappa(p))';
            % log pdf of Nakagami distribution
            logpdf_N(:,p)=nakagami_logpdf(R,m(p),omega(p));
            % log pdf of weighted combined distribution
            f_log(:,p)=logpdf_vMF(:,p)+logpdf_N(:,p)+log(alpha(p));
        end
        
        % mixture log pdf
        f_log=logsumexp(f_log,2);
        
    end
else
    error('Wrong number of Input arguments')
end

function y = vMF_logpdf(X,mu,kappa)

% X,mu,kappa
% Returns the von Mises-Fisher mixture log pdf on the unit hypersphere

d = size(X,1);
n=size(X,2);

if kappa==0
    A=log(d)+log(pi^(d/2))-gammaln(d/2+1);
    y=-A*ones(1,n);
elseif kappa>0
    c = (d/2-1)*log(kappa)-(d/2)*log(2*pi)-logbesseli(d/2-1,kappa);
    q = bsxfun(@times,mu,kappa)'*X;
    y = bsxfun(@plus,q,c');
else
    error('kappa must not be negative')
end

function y = nakagami_logpdf(X,m,om)

y=log(2)+m*(log(m)-log(om)-X.^2./om)+log(X).*(2*m-1)-gammaln(m);

function [v,b]=house(x)
% HOUSE Returns the householder transf to reduce x to b*e_n
%
% [V,B] = HOUSE(X)  Returns vector v and multiplier b so that
% H = eye(n)-b*v*v' is the householder matrix that will transform
% Hx ==> [0 0 0 ... ||x||], where  is a constant.

n=length(x);

s = x(1:n-1)'*x(1:n-1);
v= [x(1:n-1)' 1]';

if (s == 0)
    b = 0;
else
    m = sqrt(x(n)*x(n) + s);
    
    if (x(n) <= 0)
        v(n) = x(n)-m;
    else
        v(n) = -s/(x(n)+m);
    end
    b = 2*v(n)*v(n)/(s + v(n)*v(n));
    v = v/v(n);
end

function logb = logbesseli(nu,x)
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

