%% Sequential importance sampling method: Ex. 1 Ref. 2 - linear function of independent standard normal
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
Based on:

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

%% subset simulation
N  = 1000;         % Total number of samples for each level
rho = 0.1;          % Probability of each subset, chosen adaptively
% figure; hold on;

fprintf('SIS stage: \n');
[Pr, l, samplesU, samplesX, k_fin] = SIS_GM(N,rho,g,pi_pdf);  % gaussian mixture 

% exact solution
pf_ex    = normcdf(-beta);
Pf_exact = @(gg) normcdf(gg,beta,1);
gg       = 0:0.05:7;

% show p_f results
fprintf('\n***Exact Pf: %g ***', pf_ex);
fprintf('\n***SIS Pf: %g ***\n\n', Pr);

%% Plots
% plot samples
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