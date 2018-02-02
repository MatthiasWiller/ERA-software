function idx = resampling_index(w)
%% basic resampling algorithm
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
* w : unnormalized plausability weights
---------------------------------------------------------------------------
Output:
* idx : index with the highest probability
---------------------------------------------------------------------------
%}

N    = length(w);
csum = cumsum(w);
ssum = sum(w);
lim  = ssum*rand;
%
i = 1;
while csum(i) <= lim
   i = i+1;
   if i > N
      i   = 1;
      lim = ssum*rand;
   end
end
idx = i;

return;