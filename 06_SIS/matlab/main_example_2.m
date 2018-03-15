%% Subset Simulation: Ex. 2 Ref. 2 - linear function of independent exponential
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-04
* Minor changes
---------------------------------------------------------------------------
Comments:
* Compute small failure probabilities in reliability analysis of engineering systems.
* Express the failure probability as a product of larger conditional failure
 probabilities by introducing intermediate failure events.
* Use MCMC based on the modified Metropolis-Hastings algorithm for the 
 estimation of conditional probabilities.
* p0, the prob of each subset, is chosen 'adaptively' to be in [0.1,0.3]
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 100;         % number of dimensions
pi_pdf = repmat(ERADist('exponential','PAR',1),d,1);   % n independent rv

% correlation matrix
% R = eye(n);   % independent case

% object with distribution information
% pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit-state function
Ca = 140;
g  = @(x) Ca - sum(x);

%% Subset simulation
N  = 1000;         % Total number of samples for each level
rho = 0.1;         % Probability of each subset, chosen adaptively

fprintf('SIS stage: \n');
[Pr, l, samplesU, samplesX, k_fin] = SIS_GM(N,rho,g,pi_pdf);  % gaussian mixture 

% exact solution
lambda   = 1;
pf_ex    = 1 - gamcdf(Ca,d,lambda);
Pf_exact = @(gg) 1-gamcdf(Ca-gg,d,lambda);
gg       = 0:0.1:30;

% show p_f results
fprintf('\n***Exact Pf: %g ***', pf_ex);
fprintf('\n***SIS Pf: %g ***\n\n', Pf_SuS);

%%END