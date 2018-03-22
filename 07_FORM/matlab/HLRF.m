function [u_star,beta,Pf] = HLRF(H,DH,n)
%% HLRF function
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
* DH : gradient of the limit state function in the standard space
* n  : number of random variables (dim)
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

%% initialization
maxit = 1e2;
tol   = 1e-5;
u     = zeros(n,maxit);
beta  = zeros(1,maxit);

%% HLRF method
k = 1;
while true
   % 1. evaluate LSF at point u_k
   H_uk = H(u(:,k));
   
   % 2. evaluate LSF gradient at point u_k and direction cosines
   DH_uk      = DH(u(:,k));
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
beta   = beta(k);
Pf     = normcdf(-beta,0,1);

% print results
fprintf('*HLRF Method\n');
fprintf(' %g iterations... Reliability index = %g --- Failure probability = %g\n\n',k,beta,Pf);

return;