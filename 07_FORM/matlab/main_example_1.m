%% FORM using HLRF algorithm and fmincon: Ex. 1 Ref. 3 - linear function of independent standard normal
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Comment:
* The FORM method uses a first order approximation of the LSF and is 
  therefore not accurate for non-linear LSF's
---------------------------------------------------------------------------
Based on:
1."Structural reliability"
   Lemaire et al. (2009)
   Wiley-ISTE.
2."Lecture Notes in Structural Reliability"
   Straub (2016)
3."MCMC algorithms for subset simulation"
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

%% limit state function and its gradient in the original space
beta = 3.5;
g    = @(x) -sum(x)/sqrt(d) + beta;
dg   = @(x) repmat([-1/sqrt(d)],d,1);

%% Solve the optimization problem of the First Order Reliability Method

% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf] = FORM_HLRF(g, dg, pi_pdf);

% OPC 2. FORM using MATLAB fmincon
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc] = FORM_fmincon(g, pi_pdf);

% exact solution
pf_ex    = normcdf(-beta);

% show p_f results
fprintf('\n***Exact Pf: %g ***\n', pf_ex);
fprintf('***FORM HLRF Pf: %g ***\n', Pf_hlrf);
fprintf('***FORM fmincon Pf: %g ***\n\n', Pf_fmc);


%% plot HLRF results
% grid points
uu      = 0:0.05:5;
[U1,U2] = meshgrid(uu,uu);
nnu     = length(uu);
unod    = cat(2, reshape(U1,nnu^2,1),reshape(U2,nnu^2,1));
ZU      = g(unod');
ZU      = reshape(ZU,nnu,nnu);


figure;
%
hold on; pcolor(U1,U2,ZU); shading interp;
contour(U1,U2,ZU,[0 0],'r'); axis equal tight;
plot(0,0,'ro',u_star_hlrf(1),u_star_hlrf(2),'r*');   % design point in standard
line([0, u_star_hlrf(1)],[0, u_star_hlrf(2)]);       % reliability index beta
title('Standard space');

%%END