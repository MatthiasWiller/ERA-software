function h = h_calc(X, mu, Si, Pi)
%% Basic algorithm to calculate h for the likelihood ratio
%{
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* X  : input samples
* mu : [npi x d]-array of means of Gaussians in the Mixture
* Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1) 
---------------------------------------------------------------------------
Output:
* h  : parameters h (IS density)
---------------------------------------------------------------------------
%}
N=size(X,1);
if size(Pi,1)==1
    h=mvnpdf(X,mu,Si);
else
    h_pre=zeros(N,size(Pi,1));
    for q=1:size(Pi,1)
        
        h_pre(:,q)=Pi(q)*mvnpdf(X,mu(q,:),Si(:,:,q));
    end
    h=sum(h_pre,2);
end
return;
%%END