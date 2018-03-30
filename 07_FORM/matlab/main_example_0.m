%% FORM using HLRF algorithm and fmincon: FORM Example Ref. 2
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
---------------------------------------------------------------------------
%}
clear; clc; close all;

%% definition of the random variables
d      = 2;  % number of dimensions/random variables
pi_pdf = [ERADist('normal','PAR',[1,0.4]);    % x1 ~ normpdf(1,0.4)
          ERADist('exponential','PAR',[5])];  % x2 ~ exppdf(5) --mu = 1/L = 0.2 --std = 1/L = 0.2

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence


%% limit state function and its gradient in the original space
G  = @(x) x(1,:) - x(2,:);      % limit state function in original space
DG = @(x) [1;-1];               % gradient of limit state function in original space

H  = @(u) (2*u(1,:))/5 + log(1/2 - erf((2^(1/2)*u(2,:))/2)/2)/5 + 1;
DH = @(u) [ 2/5, (2^(1/2)*exp(-u(2)^2/2))/(10*pi^(1/2)*(erf((2^(1/2)*u(2))/2)/2 - 1/2))]';

%% Solve the optimization problem of the First Order Reliability Method

% OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf] = FORM_HLRF(G, DG, pi_pdf);

% OPC 2. FORM using MATLAB fmincon
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc] = FORM_fmincon(G, pi_pdf);

%% Plot HLRF results
if d == 2
  % grid points
  xx      = -1:0.1:3;
  [X1,X2] = meshgrid(xx);
  nnp     = length(xx);
  xnod    = cat(2,reshape(X1,nnp^2,1),reshape(X2,nnp^2,1));
  ZX       = G(xnod')';
  ZX       = reshape(ZX,nnp,nnp);

  uu1     = -3:0.1:1;
  uu2     = -1:0.1:3;
  [U1,U2] = meshgrid(uu1,uu2);
  nnu1    = length(uu1);
  nnu2    = length(uu2);
  unod    = cat(2, reshape(U1,nnu1*nnu2,1),reshape(U2,nnu1*nnu2,1));
  ZU      = H(unod');
  ZU      = reshape(ZU,nnu1,nnu2);


  figure;
  subplot(121); hold on; pcolor(X1,X2,ZX); shading interp;
  contour(X1,X2,ZX,[0 0],'r'); axis equal tight;
  plot(x_star_hlrf(1),x_star_hlrf(2),'r*');   % design point in original
  title('Physical space');
  %
  subplot(122); hold on; pcolor(U1,U2,ZU); shading interp;
  contour(U1,U2,ZU,[0 0],'r'); axis equal tight;
  plot(0,0,'ro',u_star_hlrf(1),u_star_hlrf(2),'r*');   % design point in standard
  line([0, u_star_hlrf(1)],[0, u_star_hlrf(2)]);       % reliability index beta
  title('Standard space');
end
%%END