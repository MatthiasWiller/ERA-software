function X = GM_sample(mu,Si,Pi,N)
%% Algorithm to draw samples from a Gaussian-Mixture (GM) distribution
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
* mu : [npi x d]-array of means of Gaussians in the Mixture
* Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1) 
* N  : number of samples to draw from the GM distribution
---------------------------------------------------------------------------
Output:
* X  : samples from the GM distribution
---------------------------------------------------------------------------
%}
if size(mu,1)==1
    X=mvnrnd(mu,Si,N);
else
    % Determine number of samples from each distribution
    z=round(Pi*N);
    
    if sum(z)~=N
        dif=sum(z)-N;
        [~,ind]=max(z);
        z(ind)=z(ind)-dif;
    end
    % Generate samples
    X=zeros(N,size(mu,2));
    ind=1;
    for p=1:size(Pi,1)
        np=z(p);
        X(ind:ind+np-1,:)=mvnrnd(mu(p,:),Si(:,:,p),np);
        ind=ind+np;
    end
end
return;
%%END