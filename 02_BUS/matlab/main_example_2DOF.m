%% BUS+SuS: Ex. 1 Ref. 1 - Parameter identification two-DOF shear building
%{
---------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-10
* Organizing the plots
Version 2017-04
* T_nataf as input in the BUS_SuS.m function
---------------------------------------------------------------------------
References:
1."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
%}
clear; clc; close all;

%% model data
% shear building data
m1 = 16.5e3;     % mass 1st story [kg]
m2 = 16.1e3;     % mass 2nd story [kg]
kn = 29.7e6;     % nominal values for the interstory stiffnesses [N/m]

%% prior PDF for X1 and X2 (product of lognormals)
mod_log_X1 = 1.3;   % mode of the lognormal 1
std_log_X1 = 1.0;   % std of the lognormal 1
mod_log_X2 = 0.8;   % mode of the lognormal 2
std_log_X2 = 1.0;   % std of the lognormal 2

% find lognormal X1 parameters
var_fun = @(mu) std_log_X1^2 - (exp(mu-log(mod_log_X1))-1)...
                              .*exp(2*mu+(mu-log(mod_log_X1)));
mu_X1  = fzero(var_fun,0);             % mean of the associated Gaussian
std_X1 = sqrt(mu_X1-log(mod_log_X1));  % std of the associated Gaussian

% find lognormal X2 parameters
var_X2 = @(mu) std_log_X2^2 - (exp(mu-log(mod_log_X2))-1)...
                             .*exp(2*mu+(mu-log(mod_log_X2)));
mu_X2  = fzero(var_X2,0);              % mean of the associated Gaussian
std_X2 = sqrt(mu_X2-log(mod_log_X2));  % std of the associated Gaussian

%% definition of the random variables
n = 2;   % number of random variables (dimensions)
% assign data: 1st variable is Lognormal
dist_x1 = ERADist('lognormal','PAR',[mu_X1, std_X1]);
% assign data: 2nd variable is Lognormal
dist_x2 = ERADist('lognormal','PAR',[mu_X2, std_X2]);

% distributions
dist_X = [dist_x1, dist_x2];
% correlation matrix
R = eye(n);   % independent case

% object with distribution information
T_nataf = ERANataf(dist_X,R);

%% likelihood function
lambda  = [1, 1]';            % means of the prediction error
i       = 9;                  % simulation level
var_eps = 0.5^(i-1);          % variance of the prediction error
f_tilde = [3.13, 9.83]';      % measured eigenfrequencies [Hz]

% shear building model 
f = @(x) shear_building_2DOF(m1, m2, kn*x(1), kn*x(2));

% modal measure-of-fit function
J = @(x) sum((lambda.^2).*(((f(x).^2)./f_tilde.^2) - 1).^2);   

% likelihood function
likelihood = @(x) exp(-J(x)/(2*var_eps));

%% find scale constant c
method = 2;
switch method
   % use MLE to find c
   case 1  
      f_start = log([mu_X1, mu_X2]);
      fun     = @(lnF) -log(likelihood(exp(lnF)) + realmin);
      options = optimset('MaxIter',1e7,'MaxFunEvals',1e7);
      MLE_ln  = fminsearch(fun,f_start,options);
      MLE     = exp(MLE_ln);   % likelihood(MLE) = 1
      c       = 1/likelihood(MLE);
      
   % some likelihood evaluations to find c
   case 2
      K  = 5e3;                  % number of samples      
      u  = randn(n,K);           % samples in standard space
      x  = T_nataf.U2X(u);       % samples in physical space
      % likelihood function evaluation
      L_eval = zeros(K,1);
      for i = 1:K
         L_eval(i) = likelihood(x(:,i));
      end
      c = 1/max(L_eval);    % Ref. 1 Eq. 34
      
   % use approximation to find c
   case 3
      fprintf('This method requires a large number of measurements\n');
      m = length(f_tilde);
      p = 0.05;
      c = 1/exp(-0.5*chi2inv(p,m));    % Ref. 1 Eq. 38
      
   otherwise
      error('Finding the scale constant c requires -method- 1, 2 or 3');
end

%% BUS-SuS
N  = 2000;       % number of samples per level
p0 = 0.1;        % probability of each subset

% run the BUS_SuS.m function
[b,samplesU,samplesX,cE] = BUS_SuS(N,p0,c,likelihood,T_nataf);

%% organize samples and show results
nsub = length(b)+1;   % number of levels + final posterior
u1p  = cell(nsub,1);   u2p  = cell(nsub,1);   u0p = cell(nsub,1);
x1p  = cell(nsub,1);   x2p  = cell(nsub,1);   pp  = cell(nsub,1);
for i = 1:nsub
   % samples in standard
   u1p{i} = samplesU.total{i}(1,:);             
   u2p{i} = samplesU.total{i}(2,:);  
   u0p{i} = samplesU.total{i}(3,:); 
   % samples in physical
   x1p{i} = samplesX.total{i}(1,:);   
   x2p{i} = samplesX.total{i}(2,:); 
   pp{i}  = samplesX.total{i}(3,:); 
end

% reference solutions
mu_exact    = 1.12;     % for x_1
sigma_exact = 0.66;     % for x_1
cE_exact    = 1.52e-3;

% show results
fprintf('\nExact model evidence = %g',cE_exact);
fprintf('\nModel evidence BUS-SuS = %g\n',cE);
fprintf('\nExact posterior mean x_1 = %g',mu_exact);
fprintf('\nMean value of x_1 = %g\n',mean(x1p{end}));
fprintf('\nExact posterior std x_1 = %g',sigma_exact);
fprintf('\nStd of x_1 = %g\n\n',std(x1p{end}));

%% plot samples
figure;
for i = 1:nsub
   subplot(2,2,i); plot(u1p{i},u2p{i},'r.'); 
   xlabel('$u_1$','Interpreter','Latex','FontSize', 18);   
   ylabel('$u_2$','Interpreter','Latex','FontSize', 18);
   set(gca,'FontSize',15); axis equal; xlim([-3 1]); ylim([-3 0]);
end
annotation('textbox', [0 0.9 1 0.1],'String', '\bf Standard space', ...
           'EdgeColor', 'none', 'HorizontalAlignment', 'center');

figure;
for i = 1:nsub
   subplot(2,2,i); plot(x1p{i},x2p{i},'b.'); 
   xlabel('$x_1$','Interpreter','Latex','FontSize', 18);
   ylabel('$x_2$','Interpreter','Latex','FontSize', 18);
   set(gca,'FontSize',15); axis equal; xlim([0 3]); ylim([0 1.5]);
end
annotation('textbox', [0 0.9 1 0.1],'String', '\bf Original space', ...
           'EdgeColor', 'none', 'HorizontalAlignment', 'center');

%%END