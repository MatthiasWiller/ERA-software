function [samples,geval] = MMA(u_j,H,b,N)
%% Modified Metropolis Algorithm
%{
---------------------------------------------------------------------------
Created by:
Junyi Jiang (junyi.jiang@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* N       : number of samples to be generated
* u_j     : seeds used to generate the new samples 100*1 double
* H       : limit state function in the standard space
* Ns      : samples generated per chain. 10 in this case
---------------------------------------------------------------------------
Output:
* samples : new samples
* geval   : limit state function evaluations of the new samples
---------------------------------------------------------------------------
References:
1."Bayesian post-processor and other enhancements of Subset Simulation for
   estimating failure probabilities in high demensions"
   Konstantin M.Zuev et al.
   Computuers and Structures 92-93 (2012) 283-296.
based on code offered by Engineering Risk Analysis Group,TUM
---------------------------------------------------------------------------
%}
%% Initialize variables
d  = size(u_j,1);      % number of uncertain parameters(dimension),100 dimensions
Ns = size(u_j,2);     
Nc = ceil(N/Ns);
%% initialization
u_jp1 = cell(Ns,1);
g_jp1 = cell(Ns,1);
%% MMA process
P = @(x)normpdf(x);        % Marginal pdf of u_j
S = @(x)unifrnd(x-1,x+1);  % Proposal pdf

% generate a candidate state epsilon
for i = 1:Ns  % 1000
    uu       = zeros(d,Nc);
    gg       = zeros(1,Nc);
    uu(:,1)  = u_j(:,i);
    gg(1)    = H(uu(:,1));
    
    for p = 1:Nc-1                              % 9 samples per chain
        xi_hat = S(uu(:,p));                    % proposal
        r      = min(1,P(xi_hat)./P(uu(:,p))); 
        xi     = zeros(d,1);                     % candidate
        for k = 1:d                              % for each dimension  
            if rand < r(k)
                xi(k)= xi_hat(k);
            else
                xi(k)= uu(k,p);
            end
        end
        % Evaluated the xi
        He = H(xi);            % LSF evaluation, call g_fun
        if He <= b              % if the sample is in failure domain
            uu(:,p+1) = xi;     % accept epsilon as new sample
            gg(p+1)= He;
        else
            uu(:,p+1) = uu(:,p);    % accept epsilon as new sample
            gg(p+1)= gg(p);
        end
    end
    u_jp1{i} = uu;
    g_jp1{i}= gg;
end

samples = cell2mat(u_jp1');
geval   = cell2mat(g_jp1');
samples = samples(:,1:N);
geval   = geval(1:N);

return;