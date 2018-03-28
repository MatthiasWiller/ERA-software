%% Sequential importance sampling method: Ex. 2 Ref. 1 - parabolic/concave limit-state function
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Based on:
1. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'),d,1);   % n independent rv

% correlation matrix
% R = eye(d);   % independent case

% object with distribution information
% pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit-state function
b=5; kappa=0.5; e=0.1;
g_fun = @(u) b - u(2,:) - kappa*(u(1,:)-e).^2;
g     = @(x) g_fun(x');

%% Subset simulation
N  = 5000;         % Total number of samples for each level
rho = 0.1;         % Probability of each subset, chosen adaptively

fprintf('SIS stage: \n');
[Pr, l, samplesU, samplesX, k_fin] = SIS_GM(N,rho,g,pi_pdf);  % gaussian mixture 

% reference solution
pf_ref   = 3.01e-3;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***SIS Pf: %g ***\n\n', Pr);

%% Plots
% plot samplesU
if d == 2
   figure; hold on;
   xx = -6:0.05:6; nnp = length(xx); [X,Y] = meshgrid(xx);
   xnod = cat(2,reshape(X',nnp^2,1),reshape(Y',nnp^2,1));
   Z    = g(xnod); Z = reshape(Z,nnp,nnp);
   contour(Y,X,Z,[0,0],'r','LineWidth',3);  % LSF
   for j = 1:l
      u_j_samples= samplesU{j};
      plot(u_j_samples(1,:),u_j_samples(2,:),'.');
   end
end

%%END