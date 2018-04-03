function [u_jk,geval,new_lambda,accrate] = aCS(N,l,b,u_j,H)
%% Adaptive conditional sampling algorithm
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de) and 
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-04
* Fixing bug in the calculation of sigma_0 (line 69)
New Version 2017-07
* Fixing allocation bugs and adaptation at the last SuS step
---------------------------------------------------------------------------
Input:
* N   : number of samples to be generated
* l   : scaling parameter lambda 
* b   : actual intermediate level
* u_j : seeds used to generate the new samples
* H   : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_jk       : next level samples
* geval      : limit state function evaluations of the new samples
* new_lambda : next scaling parameter lambda
* accrate    : acceptance rate of the method
---------------------------------------------------------------------------
Based on:
1."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
%}

%% Initialize variables
%pa = 0.1;
n  = size(u_j,1);      % number of uncertain parameters
Ns = size(u_j,2);      % number of seeds
Na = ceil(100*Ns/N);   % number of chains after which the proposal is adapted (Na = pa*Ns)

% number of samples per chain
Nchain = ones(Ns,1)*floor(N/Ns); 
Nchain(1:mod(N,Ns)) = Nchain(1:mod(N,Ns))+1;

% initialization
u_jk   = zeros(n,N);   % generated samples 
geval  = zeros(1,N);   % store lsf evaluations
acc    = zeros(1,N);   % store acceptance
mu_acc = zeros(1,floor(Ns/Na)+1);      % store acceptance
hat_a  = zeros(1,floor(Ns/Na));        % average acceptance rate of the chains
lambda = zeros(1,floor(Ns/Na)+1);      % scaling parameter \in (0,1)

%% 1. compute the standard deviation
opc = 'b';
switch opc
   case 'a'   % 1a. sigma = ones(n,1);
      sigma_0 = ones(n,1);
      
   case 'b'   % 1b. sigma = sigma_hat; (sample standard deviations)
      mu_hat  = mean(u_j,2);    % sample mean
      var_hat = zeros(n,1);     % sample std
      for i = 1:n   % dimensions
         for k = 1:Ns   % samples
            var_hat(i) = var_hat(i) + (u_j(i,k)-mu_hat(i))^2;
         end
         var_hat(i) = var_hat(i)/(Ns-1);
      end
      sigma_0 = sqrt(var_hat);
   otherwise
      error('Choose a or b');
end

%% 2. iteration
star_a    = 0.44;    % optimal acceptance rate 
lambda(1) = l;       % initial scaling parameter \in (0,1)

% a. compute correlation parameter
i         = 1;                                  % index for adaptation of lambda
sigma     = min(lambda(i)*sigma_0,ones(n,1));   % Ref. 1 Eq. 23
rho       = sqrt(1-sigma.^2);                   % Ref. 1 Eq. 24
mu_acc(i) = 0; 

% b. apply conditional sampling
for k = 1:Ns
    idx         = sum(Nchain(1:k-1))+1;  %((k-1)/pa+1);    
    acc(idx)    = 0;                % store acceptance    
    u_jk(:,idx) = u_j(:,k);         % pick a seed at random
    geval(idx)  = H(u_jk(:,idx));   % store the lsf evaluation    
    
    for t = 1:Nchain(k)-1    
        % generate candidate sample        
        v = normrnd(rho.*u_jk(:,idx+t-1), sigma);        
        %v = mvnrnd(rho.*u_jk(:,idx+t-1),diag(sigma.^2));   % n-dimensional Gaussian proposal         
        
        % accept or reject sample              
        He = H(v);
        if He <= b        
            u_jk(:,idx+t) = v;         % accept the candidate in failure region            
            geval(idx+t)  = He;        % store the lsf evaluation
            acc(idx+t)    = 1;         % note the acceptance
        else
            u_jk(:,idx+t) = u_jk(:,idx+t-1);   % reject the candidate and use the same state
            geval(idx+t)  = geval(idx+t-1);    % store the lsf evaluation    
            acc(idx+t)    = 0;                 % note the rejection
        end
    end
    % average of the accepted samples for each seed
    mu_acc(i) = mu_acc(i) + min(1, mean(acc(idx+1:idx+Nchain(k)-1))); 
    
    if mod(k,Na) == 0
        % c. evaluate average acceptance rate
        hat_a(i) = mu_acc(i)/Na;   % Ref. 1 Eq. 25
        
        % d. compute new scaling parameter
        zeta        = 1/sqrt(i);   % ensures that the variation of lambda(i) vanishes
        lambda(i+1) = exp(log(lambda(i)) + zeta*(hat_a(i)-star_a));  % Ref. 1 Eq. 26
        
        % update parameters
        sigma     = min(lambda(i+1)*sigma_0,ones(n,1));   % Ref. 1 Eq. 23
        rho       = sqrt(1-sigma.^2);                     % Ref. 1 Eq. 24
        
        % update counter
        i = i+1;
    end
end
   
% next level lambda
new_lambda = lambda(end);

% compute mean acceptance rate of all chains
accrate = mean(hat_a);

return;
%%END