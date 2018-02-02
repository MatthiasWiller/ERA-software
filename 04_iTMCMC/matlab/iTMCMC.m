function [Theta,q,S] = iTMCMC(Ns, Nb, log_likelihood, T_nataf)
%% iTMCMC function
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-04
* License check for optimization toolbox -> Use of fsolve if available in case 
  fzero does not work
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements".
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Transitional Markov Chain Monte Carlo method for Bayesian model updating,
   model class selection and model averaging".
   Jianye Ching & Yi-Chun Chen
   Journal of Engineering Mechanics. 133.7 (2007) 816-832
---------------------------------------------------------------------------
%}

%% some constants and initialization
d       = length(T_nataf.Marginals);   % number of dimensions
beta    = 2.4/sqrt(d);       % prescribed scaling factor (recommended choice)
t_acr   = 0.21/d + 0.23;     % target acceptance rate
Na      = 100;               % number of chains to adapt
thres_p = 1;                 % threshold for the c.o.v (100% recommended choice)
max_it  = 20;                % max number of iterations (for allocation)
S       = ones(max_it,1);    % space for factors S_j
q       = zeros(max_it,1);   % store tempering parameters
j       = 0;                 % initialize counter for intermediate levels

%% 1. Obtain N samples from the prior pdf and evaluate likelihood
u_j     = randn(Ns,d);          % u_0 (Nxd matrix)
theta_j = T_nataf.U2X(u_j)';    % theta_0 (transform the samples)
logL_j  = zeros(Ns,1);
for i = 1:Ns
   logL_j(i) = log_likelihood(theta_j(i,:));
end
Theta.standard{1} = u_j;           % store initial level samples
Theta.original{1} = theta_j;       % store initial level samples

%% iTMCMC
while q(j+1) < 1   % adaptively choose q
   j = j+1;
   fprintf('\niTMCMC intermediate level j = %g, with q_{j} = %g\n', j-1, q(j));
   
   % 2. Compute tempering parameter p_{j+1}
   % e   = p_{j+1}-p_{j}
   % w_j = likelihood^(e), but we are using the log_likelihood, then:
   fun = @(e) std(exp(abs(e)*logL_j)) - thres_p*mean(exp(abs(e)*logL_j));   % c.o.v equation
   [e,~,flag] = fzero(fun, 0);   
   
   % if fzero does not work try with fsolve
   if flag > 0   % OK
      e = abs(e);  % e is >= 0   
   elseif license('test','optimization_toolbox')
      option = optimset('Display','off');
      [e,~,flag] = fsolve(fun, 0, option);
      e          = abs(e);  % e is >= 0
      if flag < 0
         error('fzero and fsolve do not converge');
      end
   else 
      error('no optimization_toolbox available');
   end
   %
   if ~isnan(e)
      q(j+1) = min(1, q(j)+e);
   else
      q(j+1) = 1;
      fprintf('Variable q was set to %f, since it is not possible to find a suitable value\n',q(j+1));
   end
   
   % 3. Compute 'plausibility weights' w(theta_j) and factors 'S_j' for the evidence
   w_j  = exp((q(j+1)-q(j))*logL_j);    % [Ref. 2 Eq. 12]
   S(j) = mean(w_j);                    % [Ref. 2 Eq. 15]
   
   % 4. Metropolis resampling step to obtain N samples from f_{j+1}(theta)
   % weighted sample mean
   w_j_norm = w_j/sum(w_j);        % normalized weights
   mu_j     = u_j'*w_j_norm;   % sum inside [Ref. 2 Eq. 17]
   
   % Compute scaled sample covariance matrix of f_{j+1}(theta)
   Sigma_j = zeros(d);
   for k = 1:Ns
      tk_mu   = u_j(k,:) - mu_j';
      Sigma_j = Sigma_j + w_j_norm(k)*(tk_mu'*tk_mu);   % [Ref. 2 Eq. 17]
   end
   
   % target pdf \propto prior(\theta)*likelihood(\theta)^p(j)
   level_post = @(t,log_L) T_nataf.pdf(t).*exp(q(j+1)*log_L); % [Ref. 2 Eq. 11]
   
   % Start N different Markov chains
   msg     = fprintf('* M-H sampling...\n');
   u_c     = u_j;
   theta_c = theta_j;
   logL_c  = logL_j;
   Nadapt  = 1;
   na      = 0;
   acpt    = 0;
   for k = 1:(Ns+Nb)      
      % select index l with probability w_j
      l = resampling_index(w_j);
      
      % sampling from the proposal and evaluate likelihood
      u_star     = mvnrnd(u_c(l,:),(beta^2)*Sigma_j);
      theta_star = T_nataf.U2X(u_star)';   % transform the sample
      logL_star  = log_likelihood(theta_star);
      
      % compute the Metropolis ratio
      ratio = level_post(theta_star  ,logL_star)/...
              level_post(theta_c(l,:),logL_c(l));
      
      % accept/reject step
      if rand <= ratio 
         u_c(l,:)     = u_star;
         theta_c(l,:) = theta_star;
         logL_c(l)    = logL_star;
         acpt         = acpt+1;
      end
      if k > Nb   % (Ref. 1 Modification 2: burn-in period)
         u_j(k-Nb,:)     = u_c(l,:);
         theta_j(k-Nb,:) = theta_c(l,:);
         logL_j(k-Nb)    = logL_c(l);
      end
      
      % recompute the weights (Ref. 1 Modification 1: update sample weights)
      w_j(l) = exp((q(j+1)-q(j))*logL_c(l));
      
      % adapt beta (Ref. 1 Modification 3: adapt beta)
      na = na+1;
      if na >= Na
         p_acr  = acpt/Na;
         ca     = (p_acr - t_acr)/sqrt(Nadapt);
         beta   = beta*exp(ca);
         Nadapt = Nadapt+1;
         na     = 0;
         acpt   = 0;         
      end
   end
   fprintf(repmat('\b',1,msg));
   
   % store samples
   Theta.standard{j+1} = u_j;
   Theta.original{j+1} = theta_j;
end

% delete unnecesary data
if j < max_it
   q(j+2:end) = [];
   S(j+1:end) = [];
end

%% Compute evidence (normalization constant in Bayes' theorem)
S = prod(S);   % [Ref. 2 Eq. 17]

return;