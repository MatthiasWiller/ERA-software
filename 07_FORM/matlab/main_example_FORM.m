%% FORM example: using HLRF algorithm and fmincon
%{
---------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-04
---------------------------------------------------------------------------
References:
1."Structural reliability"
   Lemaire et al. (2009)
   Wiley-ISTE.
2."Lecture Notes in Structural Reliability"
   Straub (2016)
---------------------------------------------------------------------------
%}
clear; clc; close all;

%% definition of the random variables
n = 2;    % number of random variables
% x1 ~ normpdf(1,0.4)
% x2 ~ exppdf(5) --mu = 1/L = 0.2 --std = 1/L = 0.2
mu_x  = [1, 0.2];
std_x = [0.4, 0.2]; lambda = 5;

%% transformations X to U
syms x1 x2 u1 u2;

% Gaussian tranformation for rv x1 (Ref.1 Pag.79)
x1     = u1*std_x(1) + mu_x(1); 
fun_x1 = matlabFunction(x1);

% Independent variables tranformation for rv x2 (Ref.1 Pag.80)
Phi_u2  = 0.5*(1 + erf(u2/sqrt(2)));   % standard Gaussian CDF
Fx2_inv = @(p) -log(1-p)/lambda;       % inverse CDF exponential
x2      = Fx2_inv(Phi_u2);  
fun_x2  = matlabFunction(x2);

% copy and paste the results into H and DH
% hh  = x1-x2;          % or H  = matlabFunction(hh);
% Dhh = gradient(hh);   % or DH = matlabFunction(Dhh);

%% limit state function and its gradient in the standard space
H  = @(u) (2*u(1))/5 + log(1/2 - erf((2^(1/2)*u(2))/2)/2)/5 + 1;
DH = @(u) [ 2/5, (2^(1/2)*exp(-u(2)^2/2))/(10*pi^(1/2)*(erf((2^(1/2)*u(2))/2)/2 - 1/2))]';

%% Solve the optimization problem of the First Order Reliability Method

% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf,beta_hlrf,Pf_hlrf] = HLRF(H,DH,n);

% OPC 2. FORM using MATLAB fmincon
[u_star_fmc,beta_fmc,Pf_fmc] = opt_fmincon(H);

%% design point in original space
xx1 = fun_x1(u_star_hlrf(1));
xx2 = fun_x2(u_star_hlrf(2));
x_star_hlrf = [xx1; xx2];

%% plots HLRF results
% LSF in original and standard spaces
g = @(x1,x2) x1 - x2;
H = @(u1,u2) (2*u1)/5 + log(1/2 - erf((2^(1/2)*u2)/2)/2)/5 + 1;

% grid points
xx      = -1:0.1:3;
[X1,X2] = meshgrid(xx);
uu1     = -3:0.1:1;
uu2     = -1:0.1:3;
[U1,U2] = meshgrid(uu1,uu2);

figure;
subplot(121); hold on; pcolor(X1,X2,g(X1,X2)); shading interp;
contour(X1,X2,g(X1,X2),[0 0],'r'); axis equal tight;
plot(x_star_hlrf(1),x_star_hlrf(2),'r*');   % design point in original
title('Physical space');
%
subplot(122); hold on; pcolor(U1,U2,H(U1,U2)); shading interp;
contour(U1,U2,H(U1,U2),[0 0],'r'); axis equal tight;
plot(0,0,'ro',u_star_hlrf(1),u_star_hlrf(2),'r*');   % design point in standard
line([0, u_star_hlrf(1)],[0, u_star_hlrf(2)]);       % reliability index beta
title('Standard space');

%%END