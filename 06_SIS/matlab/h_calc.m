function h = h_calc(X, mu, Si, Pi)
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
* X  :
* mu :
* Si :
* Pi :
---------------------------------------------------------------------------
Output:
* h  : 
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