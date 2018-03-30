%% Subset Simulation: Ex. 1 Ref. 3 - convex limit-state function
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
Comments:
*Express the failure probability as a product of larger conditional failure
 probabilities by introducing intermediate failure events.
*Use MCMC based on the modified Metropolis-Hastings algorithm for the 
 estimation of conditional probabilities.
*p0, the prob of each subset, is chosen 'adaptively' to be in [0.1,0.3]
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
3."Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
%}
clear; close all; clc;

%% definition of the random variables
d      = 2;          % number of dimensions
pi_pdf = repmat(ERADist('standardnormal','PAR'),d,1);   % n independent rv

% correlation matrix
R = eye(d);   % independent case

% object with distribution information
pi_pdf = ERANataf(pi_pdf,R);    % if you want to include dependence

%% limit-state function
g = @(x) 0.1*(x(1,:)-x(2,:)).^2 - (x(1,:)+x(2,:))./sqrt(2) + 2.5;


%% subset simulation
N   = 1000;        % Total number of samples for each level
p0  = 0.1;         % Probability of each subset, chosen adaptively
alg = 'acs';       % Sampling Algorithm (either 'acs' or 'mma')

fprintf('SUBSET SIMULATION stage: \n');
[Pf_SuS,delta_SuS,b,Pf,b_sus,pf_sus,samplesU,samplesX] = SuS(N,p0,g,pi_pdf,alg);

% reference solution
pf_ref = 4.21e-3;

% show p_f results
fprintf('\n***Reference Pf: %g ***', pf_ref);
fprintf('\n***SuS Pf: %g ***\n\n', Pf_SuS);

%% Plots
% plot samples
if d == 2
   m = length(Pf);
   figure; hold on;
   xx = -6:0.05:6; nnp = length(xx); [X,Y] = meshgrid(xx);
   xnod = cat(2,reshape(X',nnp^2,1),reshape(Y',nnp^2,1));
   Z    = g(xnod'); Z = reshape(Z,nnp,nnp);
   contour(X,Y,Z,[0,0],'r','LineWidth',3);  % LSF
   for j = 1:m+1
      u_j_samples= samplesU.order{j};
      plot(u_j_samples(1,:),u_j_samples(2,:),'.');
   end
end

% Plot failure probability: Exact
figure; 
title('Failure probability estimate','Interpreter','Latex','FontSize', 20);
xlabel('Limit state function, $g$','Interpreter','Latex','FontSize', 18);   
ylabel('Failure probability, $P_f$','Interpreter','Latex','FontSize', 18);

% Plot failure probability: SuS
semilogy(b_sus,pf_sus,'r--'); hold on;  % curve
semilogy(b,Pf,'ko','MarkerSize',5);     % points
semilogy(0,Pf_SuS,'b*','MarkerSize',6);
semilogy(0,pf_ref,'ro','MarkerSize',8);
hl = legend('SuS','Intermediate levels','Pf SuS','Pf Ref.','Location','NW');
set(hl,'Interpreter','latex'); set(gca,'FontSize',18);

%%END