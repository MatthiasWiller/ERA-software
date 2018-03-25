function [h,samplesU,samplesX,cE,c,lambda] = aBUS_SuS(N,p0,log_likelihood,distr)
%% Subset simulation function for adaptive BUS
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de) and 
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
~~~~New Version 2018-01~~~~
* Bug fix in the computation of the LSF 
* Notation of the paper Ref.1
* Bug fix in the computation of the LSF 
* aCS_BUS.m function modified for Bayesian updating
---------------------------------------------------------------------------
Version 2017-07
* Bug fixes
Version 2017-04
* T_nataf is now an input
---------------------------------------------------------------------------
Input:
* N              : number of samples per level
* p0             : conditional probability of each subset
* log_likelihood : log-Likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* cE       : model evidence/marginal likelihood
* c        : scaling constant that holds 1/c >= Lmax
* lam      : scaling parameter lambda (for aCS)
---------------------------------------------------------------------------
Based on:
1."Bayesian inference with subset simulation: strategies and improvements"
   Betz et al. 
   Computer Methods in Applied Mechanics and Engineering. Accepted (2017)
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
%}
if (nargin ~= 4)
   error('Incorrect number of parameters in function "aBUS_SuS"');
end

%% log-likelihood function in standard space
% dist_p  = ERADist('uniform','PAR',[0,1]);   % uniform variable in BUS
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
   n         = length(distr.Marginals)+1;
   log_L_fun = @(u) log_likelihood(distr.U2X(u(1:end-1)));

   % if the samples are standard normal do not make any transform
   if strcmp(distr.Marginals(1).Name,'standardnormal')
      log_L_fun = @(u) log_likelihood(u(1:end-1));
   end  
   
else   % use distribution information for the transformation (independence)
   n         = length(distr)+1;
   u2x       = @(u) distr(1).icdf(normcdf(u));   % from u to x
   log_L_fun = @(u) log_likelihood(u2x(u(1:end-1)));
   
   % if the samples are standard normal do not make any transform
   if strcmp(distr(1).Name,'standardnormal')
      log_L_fun = @(u) log_likelihood(u(1:end-1));
   end
end

%% Initialization of variables
i      = 1;                   % number of conditional level
lambda = 0.6;                 % initial scaling parameter \in (0,1)
max_it = 30;                  % maximum number of iterations
%
samplesU  = cell(1,1);         % space for the samples in the standard space
log_leval = zeros(1,N);        % space for the log-likelihood evaluations
h         = zeros(max_it,1);   % space for the intermediate leveles
prob      = zeros(max_it,1);   % space for the failure probability at each level
nF        = zeros(max_it,1);   % space for the number of failure point per level

%% limit state funtion for the observation event (Ref.1 Eq.12)
gl = @(u, l, log_L) log(normcdf(u)) + l - log_L; 
% note that gl = log(p) + l(i) - log_leval; 
% where p = normcdf(u_j(end,:)) is the standard uniform variable of BUS

%% aBUS-SuS procedure
% initial log-likelihood function evaluations
u_j = randn(n,N);   % N samples from the prior distribution
fprintf('Evaluating log-likelihood function...\t');
for j = 1:N
   log_leval(j) = log_L_fun(u_j(:,j));   % evaluate likelihood
end
l = max(log_leval);   % =-log(c) (Ref.1 Alg.5 Part.3)
fprintf('Done!');

% SuS stage
h(i) = Inf;
while h(i) > 0
   % increase counter
   i = i+1;   
   
   % compute the limit state function (Ref.1 Eq.12)
   geval = gl(u_j(end,:), l, log_leval);   % evaluate LSF (Ref.1 Eq.12)
      
   % sort values in ascending order
   [~,idx] = sort(geval);
   
   % order the samples according to idx
   u_j_sort      = u_j(:,idx);
   samplesU{i-1} = u_j_sort;     % store the ordered samples
      
   % intermediate level
   h(i) = prctile(geval,p0*100); 
   fprintf('\n\n-Constant c level %g = %g\n', i-1, exp(-l));
   fprintf('-Threshold level %g = %g\n', i-1, h(i));
   
   % number of failure points
   nF(i) = sum(geval <= max(h(i),0));
   
   % assign conditional probability to the level
   if h(i) < 0
      h(i)      = 0;
      prob(i-1) = nF(i)/N;
   else
      prob(i-1) = p0;
   end
   
   % randomize the ordering of the samples (to avoid possible bias)
   seeds    = u_j_sort(:,1:nF(i));
   idx_rnd  = randperm(nF(i));
   rnd_seed = seeds(:,idx_rnd);      % non-ordered seeds
   
   % sampling process using adaptive conditional sampling (Ref.1 Alg.5 Part.4c)
   fprintf('\tMCMC sampling...\t');
   [u_j, log_leval, lambda, ~] = aCS_BUS(N, lambda, h(i), rnd_seed, log_L_fun, l, gl);
   fprintf('Done!');
   
   % update the value of the scaling constant (Ref.1 Alg.5 Part.4d)
   l_new = max(l, max(log_leval));
   h(i)  = h(i) - l + l_new;
   l     = l_new;
   fprintf('\n-New constant c level %g = %g\n', i-1, exp(-l));
   fprintf('-Modified threshold level %g = %g\n', i-1, h(i));
   
   % decrease the dependence of the samples (Ref.1 Alg.5 Part.4e)
   p          = unifrnd( zeros(1,N), min(ones(1,N),exp(log_leval -l + h(i))) );
   u_j(end,:) = norminv(p);   % to the standard space
end

% number of intermediate levels
m = i;   

% store final posterior samples
samplesU{m} = u_j;

% delete unnecesary data
if m < max_it
   h(m+1:end)  = [];    
   prob(m:end) = [];
end

%% acceptance probability and model evidence (Ref.1 Alg.5 Part.6and7)
p_acc = prod(prob);
c     = 1/exp(l);       
cE    = p_acc*exp(l);   % l = -log(c)

%% transform the samples to the physical/original space
samplesX = cell(m,1);
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
   if strcmp(distr.Marginals(1).Name,'standardnormal')
       for i = 1:m
         p           = normcdf(samplesU{i}(end,:));
         samplesX{i} = [samplesU{i}(1:end-1,:); p];
       end
   else
      for i = 1:m
         p           = normcdf(samplesU{i}(end,:));
         samplesX{i} = [distr.U2X(samplesU{i}(1:end-1,:)); p];
      end
   end
else
   if strcmp(distr(1).Name,'standardnormal')
       for i = 1:m
         p           = normcdf(samplesU{i}(end,:));
         samplesX{i} = [samplesU{i}(1:end-1,:); p];
      end
   else
      for i = 1:m
         p           = normcdf(samplesU{i}(end,:));
         samplesX{i} = [u2x(samplesU{i}(1:end-1,:)); p];
      end
   end
end

return;
%%END