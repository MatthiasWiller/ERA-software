function [u_star,x_star,beta,Pf] = FORM_fmincon(g, distr)
%% optimization using the fmincon function
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
Input:
* g     : limit state function in the original space
* distr : ERANataf-Object containing the distribution
---------------------------------------------------------------------------
Output:
* u_star : design point in the standard space
* x_star : design point in the original space
* beta   : reliability index
* Pf     : probability of failure
---------------------------------------------------------------------------
References:
1."Structural reliability"
   Lemaire et al. (2009)
   Wiley-ISTE.
---------------------------------------------------------------------------
%}

%% initial check if there exists a Nataf object
if ~(any(strcmp('Marginals',fieldnames(distr))) == 1)   % use Nataf transform (dependence)
	return;
end

d =  length(distr.Marginals); 

%% objective function
dist_fun = @(u) norm(u);

%% parameters of the fmincon function
u0  = repmat([0.01],d,1);  % initial search point
A   = [];         % linear equality constraints
b   = [];         % linear equality constraints
Aeq = [];         % linear inequality constraints
beq = [];         % linear inequality constraints
lb  = [];         % lower bound constraints
ub  = [];         % upper bound constraints

% nonlinear constraint: H(u) <= 0
H      = @(u) g(distr.U2X(u));
lsfcon = @(u) deal([], H(u'));

%% use fmincon
options = optimoptions('fmincon','Display','off','Algorithm','sqp');
[u_star,beta,~,output] = fmincon(dist_fun,u0,A,b,Aeq,beq,lb,ub,lsfcon,options);

iter = output.iterations;
alg  = output.algorithm;

% compute design point in orignal space and failure probability
x_star = distr.U2X(u_star);
Pf     = normcdf(-beta);

% print results
fprintf('*fmincon with %s Method\n',alg);
fprintf(' %g iterations... Reliability index = %g --- Failure probability = %g\n\n',iter,beta,Pf);

return;