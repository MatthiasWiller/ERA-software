function [Pf_SuS,delta_SuS,b,Pf,b_line,Pf_line,samplesU] = SuS(N,p0,g_fun,distr)
%% Subset Simulation function (standard Gaussian space)
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-10
* Computing the correct failure samples
Version 2017-04
* Minor changes
---------------------------------------------------------------------------
Input:
* N         : Number of samples per level
* p0        : Conditional probability of each subset
* g_fun     : limit state function
* distr     : Nataf distribution object or
              marginal distribution object of the input variables
---------------------------------------------------------------------------
Output:
* Pf_SuS    : failure probability estimate of subset simulation
* delta_SuS : coefficient of variation estimate of subset simulation 
* b         : intermediate failure levels
* Pf        : intermediate failure probabilities
* b_line    : limit state function values
* Pf_line   : failure probabilities corresponding to b_line
* samplesU  : samples in the Gaussian standard space for each level
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2. "MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
%}

%% initial check if there exists a Nataf object
if any(strcmp('Marginals',fieldnames(distr))) == 1   % use Nataf transform (dependence)
   n = length(distr.Marginals);    % number of random variables (dimension)
   g = @(u) g_fun(distr.U2X(u));   % LSF in standard space
   
   % if the samples are standard normal do not make any transform
   if strcmp(distr.Marginals(1).Name,'standardnormal')
      g = g_fun;
   end
else   % use distribution information for the transformation (independence)
   n   = length(distr);                    % number of random variables (dimension)
   u2x = @(u) distr(1).icdf(normcdf(u));   % from u to x
   g   = @(u) g_fun(u2x(u));               % LSF in standard space
   
   % if the samples are standard normal do not make any transform
   if strcmp(distr(1).Name,'standardnormal')
      g = g_fun;
   end
end

%% Initialization of variables and storage
j      = 1;                % initial conditional level
Nc     = N*p0;             % number of markov chains
Ns     = 1/p0;             % number of samples simulated from each Markov chain
lambda = 0.6;              % recommended initial value for lambda
max_it = 20;               % estimated number of iterations
%
geval = zeros(1,N);        % space for the LSF evaluations
gsort = zeros(max_it,N);   % space for the sorted LSF evaluations
delta = zeros(max_it,1);   % space for the coefficient of variation
Nf    = zeros(max_it,1);   % space for the number of failure point per level
prob  = zeros(max_it,1);   % space for the failure probability at each level
b     = zeros(max_it,1);   % space for the intermediate leveles

%% SuS procedure
% initial MCS stage
fprintf('Evaluating performance function:\t');
u_j = randn(n,N);     % samples in the standard space
for i = 1:N
   geval(i) = g(u_j(:,i));
   if geval(i) <= 0
      Nf(j) = Nf(j)+1;    % number of failure points
   end
end
fprintf('OK! \n');

% SuS stage
while true 
   % sort values in ascending order
   [gsort(j,:),idx] = sort(geval);
   
   % order the samples according to idx
   u_j_sort = u_j(:,idx);
   samplesU.order{j} = u_j_sort;   % store the ordered samples

   % intermediate level
   b(j) = prctile(gsort(j,:),p0*100);
   
   % number of failure points in the next level
   nF = sum(gsort(j,:) <= max(b(j),0));

   % assign conditional probability to the level
   if b(j) <= 0
      b(j)    = 0;
      prob(j) = nF/N;
   else
      prob(j) = p0;      
   end
   fprintf('\n-Threshold intermediate level %g = %g \n', j-1, b(j));

   % compute coefficient of variation
   if j == 1
      delta(j) = sqrt(((1-p0)/(N*p0)));   % cov for p(1): MCS (Ref. 2 Eq. 8)
   else
      I_Fj     = reshape(geval <= b(j),Ns,Nc);         % indicator function for the failure samples
      p_j      = (1/N)*sum(I_Fj(:));                   % ~=p0, sample conditional probability
      gamma    = corr_factor(I_Fj,p_j,Ns,Nc);          % corr factor (Ref. 2 Eq. 10)
      delta(j) = sqrt( ((1-p_j)/(N*p_j))*(1+gamma) );  % coeff of variation(Ref. 2 Eq. 9)
   end
   
   % select seeds
   samplesU.seeds{j} = u_j_sort(:,1:nF);       % store ordered level seeds
   
   % randomize the ordering of the samples (to avoid bias)
   idx_rnd   = randperm(nF);
   rnd_seeds = samplesU.seeds{j}(:,idx_rnd);   % non-ordered seeds
   
   % sampling process using adaptive conditional sampling
   %[u_j,geval,lambda,~] = aCS(N,lambda,b(j),rnd_seeds,g);
   [u_j,geval,] = MMA(rnd_seeds,g,b(j),N);
   
   % next level
   j = j+1;   
   
   if b(j-1) <= 0 || j-1 == max_it
      break;
   end
end
m = j-1;
samplesU.order{j} = u_j;  % store final failure samples (non-ordered)

% delete unnecesary data
if m < max_it
   gsort(m+1:end,:) = [];
   prob(m+1:end)    = [];
   b(m+1:end)       = [];
   delta(m+1:end)   = [];
end

%% probability of failure
% failure probability estimate
Pf_SuS = prod(prob);   % or p0^(m-1)*(Nf(m)/N);

% coeficient of variation estimate
delta_SuS = sqrt(sum(delta.^2));   % (Ref. 2 Eq. 12)

%% Pf evolution 
Pf           = zeros(m,1);
Pf(1)        = p0;
Pf_line(1,:) = linspace(p0,1,Nc);
b_line(1,:)  = prctile(gsort(1,:),Pf_line(1,:)*100);
for i = 2:m
   Pf(i)        = Pf(i-1)*p0;
   Pf_line(i,:) = Pf_line(i-1,:)*p0;
   b_line(i,:)  = prctile(gsort(i,:),Pf_line(1,:)*100);
end
Pf_line = sort(Pf_line(:));
b_line  = sort(b_line(:));

return;
%%END