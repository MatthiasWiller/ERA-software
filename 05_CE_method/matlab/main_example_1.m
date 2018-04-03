%% Cross entropy method: Ex. 1 Ref. 2 - linear function of independent standard normal
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
Comments:
* The CE-method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
* General convergence issues can be observed with linear LSFs.
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling 
   using Gaussian densities revisited"
   Geyer et al.
   Engineering Risk Analysis Group, TUM (Sep 2017)
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'),d,1);   % n independent rv

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit-state function
beta = 3.5;
g    = @(x) -sum(x)/sqrt(d) + beta;

%% CE-method
N      = 1000;    % Total number of samples for each level
rho    = 0.1;     % Cross-correlation coefficient for conditional sampling
k_init = 3;       % Initial number of distributions in the Mixture Model (GM/vMFNM)

fprintf('CE-based IS stage: \n');
[Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_SG(N,rho,g,pi_pdf);     % single gaussian 
% [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_GM(N,rho,g,pi_pdf,k_init);    % gaussian mixture
% [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin] = CEIS_vMFNM(N,rho,g,pi_pdf,k_init); % adaptive vMFN mixture


% exact solution
pf_ex    = normcdf(-beta);
Pf_exact = @(gg) normcdf(gg,beta,1);
gg       = 0:0.05:7;

% show p_f results
fprintf('\n***Exact Pf: %g ***', pf_ex);
fprintf('\n***CEIS Pf: %g ***\n\n', Pr);

%% Plots
% plot samplesU
if d == 2
   figure; hold on;
   xx = 0:0.05:5; nnp = length(xx); [X,Y] = meshgrid(xx);
   xnod = cat(2,reshape(X',nnp^2,1),reshape(Y',nnp^2,1));
   Z    = g(xnod'); Z = reshape(Z,nnp,nnp);
   contour(X,Y,Z,[0,0],'r','LineWidth',3);  % LSF
   for j = 1:l
      u_j_samples= samplesU{j};
      plot(u_j_samples(1,:),u_j_samples(2,:),'.');
   end
end

%%END