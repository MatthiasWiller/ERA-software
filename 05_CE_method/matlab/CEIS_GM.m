function [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_GM(N,rho,g_fun,distr)
%% Cross entropy-based importance sampling with Gaussian Mixture
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
* N     : Number of samples per level
* rho   : cross-correlation coefficient for conditional sampling
* g_fun : limit state function
* distr : Nataf distribution object or
          marginal distribution object of the input variables
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* l         : total number of levels
* N_tot     : total number of samples
* gamma_hat : gamma_ik probability of sample i belonging to distribution k 
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of Gaussians in the mixture
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling 
   using Gaussian densities revisited"
   Geyer et al.
   Engineering Risk Analysis Group, TUM (Sep 2017)
---------------------------------------------------------------------------
%}

%% initial check if there exists a Nataf object
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
   dim = length(distr.Marginals);    % number of random variables (dimension)
   g   = @(u) g_fun(distr.U2X(u));   % LSF in standard space
   
   % if the samples are standard normal do not make any transform
   if strcmp(distr.Marginals(1).Name,'standardnormal')
      g = g_fun;
   end
else   % use distribution information for the transformation (independence)
   dim = length(distr);                    % number of random variables (dimension)
   u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x
   g   = @(u) g_fun(u2x(u));               % LSF in standard space
   
   % if the samples are standard normal do not make any transform
   if strcmp(distr(1).Name,'standardnormal')
      g = g_fun;
   end
end

%% Initialization of variables and storage
j      = 0;                % initial level
max_it = 100;              % estimated number of iterations
N_tot  = 0;                % total number of samples
k      = 1;                % number of Gaussians in mixture

% Definition of parameters of the random variables (uncorrelated standard normal)
mu_init = zeros(1,dim);   % ...
Si_init = eye(dim);       % ...
Pi_init = 1;              % ...
%
gamma_hat = zeros(max_it+1,1); % space for ...
samplesU = cell(1,1);

%% CE procedure
% initializing parameters
gamma_hat(1) = 1;
mu_hat       = mu_init;
Si_hat       = Si_init;
Pi_hat       = Pi_init;

%% Iteration
for j = 1:max_it
  % Generate samples
  X = GM_sample(mu_hat, Si_hat, Pi_hat, N);
  samplesU{j} = X';

  % Count generated samples
  N_tot = N_tot+N;

  % Evaluation of the limit state function
  geval=g(X');

  % Calculating h for the likelihood ratio
  h=h_calc(X,mu_hat,Si_hat,Pi_hat);

  % Check convergence
  if gamma_hat(j) == 0
      k_fin = k;
      break
  end

  % obtaining estimator gamma
  gamma_hat(j+1) = max(0,prctile(geval, rho*100));
  disp(num2str(gamma_hat(j+1)));

  % Indicator function
  I=(geval<=gamma_hat(j+1));

  % Likelihood ratio
  W=mvnpdf(X,zeros(1,dim),eye(dim))./h;

  % Parameter update
  nGM=3;
  [mu, si, pi] = EMGM(X(I,:)',W(I),nGM);

  % Assigning the variables with updated parameters
  mu_hat=mu';
  Si_hat=si;
  Pi_hat=pi';
  k=length(pi);

end

% needed steps
l = j; k_fin = k;

%% Calculation of the Probability of failure
W_final = mvnpdf(X,zeros(1,dim),eye(dim))./h;
I_final = (geval<=0);
Pr      = 1/N*sum(I_final*W_final);

%% transform the samples to the physical/original space
samplesX = cell(l,1);
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
   if strcmp(distr.Marginals(1).Name,'standardnormal')
       for i = 1:l
         samplesX{i} = samplesU{i}(:,:);
       end
   else
      for i = 1:l
         samplesX{i} = distr.U2X(samplesU{i}(:,:));
      end
   end
else
   if strcmp(distr(1).Name,'standardnormal')
       for i = 1:l
         samplesX{i} = samplesU{i}(:,:);
      end
   else
      for i = 1:l
         samplesX{i} = u2x(samplesU{i}(:,:));
      end
   end
end

return;
%%END