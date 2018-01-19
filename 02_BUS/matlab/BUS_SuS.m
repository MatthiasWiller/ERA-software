function [b,samplesU,samplesX,cE] = BUS_SuS(N,p0,c,likelihood,T_nataf)
%% Subset simulation function adapted for BUS (likelihood input)
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-10
* Fixing a bug in the computation of the posterior samples
Version 2017-04
* T_nataf is now an input
---------------------------------------------------------------------------
Input:
* N         : number of samples per level
* p0        : conditional probability of each subset
* c         : scaling constant of the BUS method
* likelihood: likelihood function of the problem at hand
* T_nataf   : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* b        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* cE       : model evidence/marginal likelihood
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
%}
if (nargin ~= 5)
   error('Incorrect number of parameters in function "BUS_SuS"');
end

%% add p Uniform variable of BUS
n = length(T_nataf.Marginals)+1;        % number of parameters (dimension)
%dist_p  = ERADist('uniform','PAR',[0,1]);     % uniform variable in BUS

%% limit state function in the standard space
H = @(u) u(end) - norminv(c*likelihood(T_nataf.U2X(u(1:end-1))));

%% Initialization of variables
j      = 1;                         % number of conditional level
lambda = 0.6;                       % initial scaling parameter \in (0,1)
max_it = 20;                        % maximum number of iterations
%
geval = zeros(1,N);        % space for the LSF evaluations
gsort = zeros(max_it,N);   % space for the sorted LSF evaluations
Nf    = zeros(max_it,1);   % space for the number of failure point per level
b     = zeros(max_it,1);   % space for the intermediate leveles
prob  = zeros(max_it,1);   % space for the failure probability at each level

%% SuS procedure
% initial MCS step
fprintf('Evaluating performance function:\t')
u_j = randn(n,N);     % samples in the standard space
for i = 1:N
   geval(i) = H(u_j(:,i));    % limit state function in standard (Ref. 2 Eq. 21)
   if geval(j,i) <= 0
      Nf(j) = Nf(j)+1;
   end
end
fprintf('OK! \n');

% SuS stage
while true
   % sort values in ascending order
   [gsort(j,:),idx] = sort(geval);
   
   % order the samples according to idx
   u_j_sort = u_j(:,idx);
   samplesU.total{j} = u_j_sort;   % store the ordered samples
   
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
   
   % select seeds
   samplesU.seeds{j} = u_j_sort(:,1:nF);       % store ordered level seeds
   
   % randomize the ordering of the samples (to avoid bias)
   idx_rnd   = randperm(nF);
   rnd_seeds = samplesU.seeds{j}(:,idx_rnd);   % non-ordered seeds
   
   % sampling process using adaptive conditional sampling
   [u_j,geval,lambda,~] = aCS(N,lambda,b(j),rnd_seeds,H);
   
   % next level
   j = j+1;   
   
   if b(j-1) <= 0 || j-1 == max_it
      break;
   end
end
m = j-1;
samplesU.total{j} = u_j;  % store final posterior samples (non-ordered)

% delete unnecesary data
if m < max_it
   b(m+1:end)       = [];
   prob(m+1:end)    = [];
end

%% acceptance probability and evidence
p_acc = prod(prob);
cE    = p_acc/c;

%% transform the samples to the physical (original) space
for i = 1:m+1
   %p = dist_p.icdf(normcdf(samplesU.total{i}(end,:))) is the same as:
   p = normcdf(samplesU.total{i}(end,:));
   samplesX.total{i} = [T_nataf.U2X(samplesU.total{i}(1:end-1,:)); p];  
end

return;