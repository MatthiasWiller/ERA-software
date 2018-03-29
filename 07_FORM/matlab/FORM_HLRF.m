function [u_star,x_star,beta,Pf] = FORM_HLRF(g,dg,distr)
%% HLRF function
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
Input:
* g     : limit state function in the original space
* dg    : gradient of the limit state function in the standard space
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

n = length(distr.Marginals);    % number of random variables (dimension)

%% initialization
maxit = 1e2;
tol   = 1e-5;
u     = zeros(n,maxit);
beta  = zeros(1,maxit);

%% HLRF method
k = 1;
while true
   % 0. Get x and Jacobi from u (important for transformation)
   [xk, J] = distr.U2X(u(:,k), 'Jac');
   
   % 1. evaluate LSF at point u_k
   H_uk = g(xk);
   
   % 2. evaluate LSF gradient at point u_k and direction cosines
   DH_uk      = J\dg(xk);
   norm_DH_uk = norm(DH_uk);
   alpha      = DH_uk/norm_DH_uk;
   
   % 3. calculate beta
   beta(k) = -u(:,k)'*alpha + (H_uk/norm_DH_uk);
   
   % 4. calculate u_{k+1}
   u(:,k+1) = -beta(k)*alpha;
   
   % next iteration
   if (norm(u(:,k+1)-u(:,k)) <= tol)  || (k == maxit)
      break;
   else
      k = k+1;
   end
end

% delete unnecessary data
u(:,k+2:end) = [];

% compute design point, reliability index and Pf
u_star = u(:,end);
x_star = distr.U2X(u_star);
beta   = beta(k);
Pf     = normcdf(-beta,0,1);

% print results
fprintf('*FORM Method\n');
fprintf(' %g iterations... Reliability index = %g --- Failure probability = %g\n\n',k,beta,Pf);

return;