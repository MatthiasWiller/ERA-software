%% FORM using HLRF algorithm and fmincon: Ex. 3 Ref. 3 - series system reliability problem
%{
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
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
3. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
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
g     = @(x) min([0.1.*(x(:,1)-x(:,2)).^2-(x(:,1)+x(:,2))./sqrt(2)+3,0.1.*(x(:,1)-x(:,2)).^2+(x(:,1)+x(:,2))./sqrt(2)+3,x(:,1)-x(:,2)+7./sqrt(2),x(:,2)-x(:,1)+7./sqrt(2)],[],2);
g_fun = @(x) g(x');
% Due to the min()-function the gradient of g is not included in this
% example. Therefore, we cannot apply the HLRF algorithm here.

%% Solve the optimization problem of the First Order Reliability Method

% OPC 1. FORM using MATLAB fmincon
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc] = FORM_fmincon(g_fun, pi_pdf);

% reference solution
pf_ref = 2.2e-3;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***FORM fmincon Pf: %g ***\n\n', Pf_fmc);

%% Plot
% grid points
uu      = -6:0.05:6;
[U1,U2] = meshgrid(uu,uu);
nnu     = length(uu);
unod    = cat(2, reshape(U1,nnu^2,1),reshape(U2,nnu^2,1));
ZU      = g(unod);
ZU      = reshape(ZU,nnu,nnu);

figure;
%
hold on; pcolor(U1,U2,ZU); shading interp;
contour(U1,U2,ZU,[0 0],'r'); axis equal tight;
plot(0,0,'ro',u_star_fmc(1),u_star_fmc(2),'r*');   % design point in standard
line([0, u_star_fmc(1)],[0, u_star_fmc(2)]);       % reliability index beta
title('Standard space');

%%END