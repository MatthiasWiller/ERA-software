function [Pr, l_tot, k_fin] = SIS_GM(N,rho,g_fun,distr)
%% Sequential importance sampling
%{
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
* N         : Number of samples per level
* rho       : 
* g_fun     : limit state function
* distr     : Nataf distribution object or
              marginal distribution object of the input variables
---------------------------------------------------------------------------
Output:

---------------------------------------------------------------------------
Based on:

---------------------------------------------------------------------------
%}

%% initial check if there exists a Nataf object
% if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
%    dim = length(distr.Marginals);    % number of random variables (dimension)
%    g   = @(u) g_fun(distr.U2X(u));   % LSF in standard space
%    
%    % if the samples are standard normal do not make any transform
%    if strcmp(distr.Marginals(1).Name,'standardnormal')
%       g = g_fun;
%    end
% else   % use distribution information for the transformation (independence)
%    dim = length(distr);                    % number of random variables (dimension)
%    u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x
%    g   = @(u) g_fun(u2x(u));               % LSF in standard space
%    
%    % if the samples are standard normal do not make any transform
%    if strcmp(distr(1).Name,'standardnormal')
%       g = g_fun;
%    end
% end
dim = 2;
g = g_fun;
nsamlev = N;
nchain = nsamlev*rho;
lenchain=nsamlev/nchain;
numsim = 0;

% burn-in
burn = 0;

% tolerance of COV of weights
tolCOV = 1.5;

% initialize samples
uk=zeros(nsamlev,dim);


%% Initialization of variables and storage
max_it = 100;              % estimated number of iterations

%%% Step 1
% Perform the first Monte Carlo simulation
for k=1:nsamlev 

   % Do the simulation (create array of random numbers)
   u = randn(dim,1);
   uk(k,:) = u;
   gk(k) = g(u');  
   numsim = numsim+1;
end

% set initial subset and failure level
m = 1;
gmu = mean(gk);
sigmak(m) = 50*gmu;

%% Iteration
for j = 1:max_it
  
  %%% Step 2 and 3
  % compute sigma and weights
  if m == 1
      sigma2      = fminbnd(@(x)abs(std(normcdf(-gk/x))/mean(normcdf(-gk/x))-tolCOV),0,10.0*gmu);
      sigmak(m+1) = sigma2;
      wk          = normcdf(-gk/sigmak(m+1));
  else     
      sigma2      = fminbnd(@(x)abs(std(normcdf(-gk/x)./normcdf(-gk/sigmak(m)))/mean(normcdf(-gk/x)./normcdf(-gk/sigmak(m)))-tolCOV),0,sigmak(m));
      sigmak(m+1) = sigma2;
      wk          = normcdf(-gk/sigmak(m+1))./normcdf(-gk/sigmak(m));
  end

  %%% Step 4
  % compute estimate of expected w
  Sk(m) = mean(wk);

  % Exit algorithm if no convergence is achieved
  if Sk(m)==0    
    pfk=0;
    break
  end
  
  % compute normalized weights
  wnork = wk./Sk(m)/nsamlev;

  % fit Gaussian Mixture
  nGM = 2;
  dist = EMGM(uk',wnork',nGM);
  
  %%% Step 5
  % resample
  ind = randsample(nsamlev,nchain,true,wnork);

  % seeds for chains
  gk0 = gk(ind);
  uk0 = uk(ind,:);

  %%% Step 6
  % perform M-H
  count = 0;

  % initialize chain acceptance rate
  alphak = zeros(nchain,1);    

  % delete previous samples
  gk = [];
  uk = [];
  
  for k=1:nchain
    % set seed for chain
    u0 = uk0(k,:);
    g0 = gk0(k);

    for j=1:lenchain+burn                       
      count = count+1;
            
      if j == burn+1
         count = count-burn; 
      end
            
      % get candidate sample from conditional normal distribution 
      indw = randsample(length(dist.pi),1,true,dist.pi);
      ucand = mvnrnd(dist.mu(:,indw),dist.si(:,:,indw));
      %ucand = muf + (Acov*randn(dim,1))';
            
      % Evaluate limit-state function              
      gcand = g(ucand);             

      numsim = numsim+1;

      % compute acceptance probability
      pdfn = 0;
      pdfd = 0;
      for i = 1:length(dist.pi)
          pdfn = pdfn+dist.pi(i)*mvnpdf(u0',dist.mu(:,i),dist.si(:,:,i));
          pdfd = pdfd+dist.pi(i)*mvnpdf(ucand',dist.mu(:,i),dist.si(:,:,i));

      end
            
      alpha     = min(1,normcdf(-gcand/sigmak(m+1))*prod(normpdf(ucand))*pdfn/normcdf(-g0/sigmak(m+1))/prod(normpdf(u0))/pdfd);
      alphak(k) = alphak(k)+alpha/(lenchain+burn);

      % check if sample is accepted
      uhelp = rand;
      if uhelp <= alpha
        uk(count,:) = ucand;
        gk(count)   = gcand;                
        u0 = ucand;
        g0 = gcand;
      else
        uk(count,:) = u0;
        gk(count)   = g0;
      end
                                
    end
  
  end

  uk = uk(1:nsamlev,:);
  gk = gk(1:nsamlev);    

  % compute mean acceptance rate of all chains in level m
  accrate(m) = mean(alphak);
  COV_Sl = std((gk < 0)./normcdf(-gk/sigmak(m+1)))/mean((gk < 0)./normcdf(-gk/sigmak(m+1)))
  if COV_Sl < 0.01   
      break;
  end                
    
  m = m+1;
end

k_fin=length(dist.pi);

const=prod(Sk);

% probability of failure
Pr = mean((gk < 0)./normcdf(-gk/sigmak(m+1)))*const;

accfin = accrate(m);
l_tot=m+1;


return;
%%END