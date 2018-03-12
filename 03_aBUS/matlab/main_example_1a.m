%% aBUS+SuS: Ex. 1 Ref. 1 - Parameter identification two-DOF shear building
%{
---------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-04
* T_nataf as input in the aBUS_SuS.m function
---------------------------------------------------------------------------
References:
1."Bayesian inference with subset simulation: strategies and improvements"
   Betz et al.
   Submitted (2017)
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
%}
clear; clc; close all;

%% definition of the random variables
n = 1;   % number of random variables (dimensions)
% assign data: 1st variable is Lognormal
dist_x1 = ERADist('standardnormal','MOM',[0, 1]);

% distributions
dist_X = repmat(dist_x1,n,1);
% correlation matrix
R = eye(n);   % independent case

% object with distribution information
T_nataf = ERANataf(dist_X,R);

%% likelihood function
mu_l    = 3.0;
sigma_l = 0.3;
likelihood     = @(x) normpdf(x,mu_l,sigma_l);
log_likelihood = @(x) log(likelihood(x) + realmin);   % realmin to avoid Inf values in log(0)

%% aBUS-SuS
N  = 1000;       % number of samples per level
p0 = 0.1;        % probability of each subset

% run the aBUS_SuS.m function
[h,samplesU,samplesX,cE,c,lamda_new] = aBUS_SuS(N,p0,log_likelihood,T_nataf);

%% take the samples


%% reference and aBUS solutions
mu_exact    = 2.75;      % for x_1
sigma_exact = 0.287;     % for x_1
cE_exact    = 6.16e-3;
L_max       = 1.33;

% show results
fprintf('\nExact model evidence = %g',cE_exact);
fprintf('\nModel evidence aBUS-SuS = %g\n',cE);
fprintf('\nExact maximum likelihood = %g',L_max);
fprintf('\nMaximum likelihood aBUS-SuS = %g\n',1/c);
fprintf('\nExact posterior mean x_1 = %g',mu_exact);
fprintf('\nMean value of x_1 = %g\n',mean(samplesU{end}(1,:)));
fprintf('\nExact posterior std x_1 = %g',sigma_exact);
fprintf('\nStd of x_1 = %g\n\n',std(samplesU{end}(1,:)));

%%END