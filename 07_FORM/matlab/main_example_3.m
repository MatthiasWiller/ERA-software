%% FORM using HLRF algorithm and fmincon: Ex. 1 Ref. 3 - convex limit-state function
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
g  = @(x) 0.1*(x(1,:)-x(2,:)).^2 - (x(1,:)+x(2,:))./sqrt(2) + 2.5;
dg = @(x) [0.2*(x(1,:)-x(2,:)) - 1/sqrt(2); -0.2*(x(1,:)-x(2,:)) - 1/sqrt(2)];


%% Solve the optimization problem of the First Order Reliability Method

% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf] = FORM_HLRF(g, dg, pi_pdf);

% OPC 2. FORM using MATLAB fmincon
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc] = FORM_fmincon(g, pi_pdf);

% reference solution
pf_ref = 4.21e-3;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***FORM HLRF Pf: %g ***', Pf_hlrf);
fprintf('\n***FORM fmincon Pf: %g ***\n\n', Pf_fmc);

%% plot HLRF results
% grid points
uu      = -6:0.05:6;
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