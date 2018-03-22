function [u_star,beta,Pf] = opt_fmincon(H)
%% optimization using the fmincon function
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
Input:
* H  : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_star : design point in the standard space
* beta   : reliability index
* Pf     : probability of failure
---------------------------------------------------------------------------
References:
1."Structural reliability"
   Lemaire et al. (2009)
   Wiley-ISTE.
---------------------------------------------------------------------------
%}

%% objective function
dist_fun = @(u) norm(u);

%% parameters of the fmincon function
u0  = [0.1,0.1];  % initial search point
A   = [];         % linear equality constraints
b   = [];         % linear equality constraints
Aeq = [];         % linear inequality constraints
beq = [];         % linear inequality constraints
lb  = [-5,-5];    % lower bound constraints
ub  = [5,5];      % upper bound constraints

% nonlinear constraint: H(u) <= 0
lsfcon = @(u) deal([], H(u));

%% use fmincon
options = optimoptions('fmincon','Display','off','Algorithm','sqp');
[u_star,beta,~,output] = fmincon(dist_fun,u0,A,b,Aeq,beq,lb,ub,lsfcon,options);

iter = output.iterations;
alg  = output.algorithm;

% failure probability
Pf = normcdf(-beta);

% print results
fprintf('*fmincon with %s Method\n',alg);
fprintf(' %g iterations... Reliability index = %g --- Failure probability = %g\n\n',iter,beta,Pf);

return;