function X = GM_sample(mu,Si,Pi,N)
%% Basic algorithm
%{
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
---------------------------------------------------------------------------
Input:
* mu :
* Si :
* Pi :
* N  :
---------------------------------------------------------------------------
Output:
* X  : 
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