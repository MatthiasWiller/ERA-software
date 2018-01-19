%% Example file: Use of ERADist and ERANataf
%{
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer (s.geyer@tum.de), 
Iason Papaioannou and Felipe Uribe
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-04
---------------------------------------------------------------------------
References:
1. ERADist-ERANataf documentation
---------------------------------------------------------------------------
%}
clear; clc; close all;

%% Examples
example = 1;
switch example
   case 1
      M(1) = ERADist('normal','PAR',[4,2]);
      M(2) = ERADist('gumbel','MOM',[1,2]);
      M(3) = ERADist('exponential','PAR',4);
      Rho  = [ 1.0 0.5 0.5;
               0.5 1.0 0.5;
               0.5 0.5 1.0 ];   % correlation matrix
   case 2
      M(1) = ERADist('rayleigh','PAR',1);
      M(2) = ERADist('gumbel','MOM',[0,1]);
      Rho  = [ 1.0 0.6;
               0.6 1.0  ];   % correlation matrix
   case 3
      M(1) = ERADist('gamma','PAR',[2,1]);
      M(2) = ERADist('chisquare','PAR',5);
      Rho  = [ 1.0 0.5;
               0.5 1.0  ];   % correlation matrix
   case 4
      M(1) = ERADist('gamma','PAR',[1,2]);
      M(2) = ERADist('weibull','PAR',[4,5]);
      Rho  = [ 1.0 0.1  
               0.1 1.0 ];  
end

%% applying Nataf transformation
T_Nataf = ERANataf(M,Rho);

% generation of random samples from the joint distribution
N = 1e3;                 % number of samples
X = T_Nataf.random(N);

% samples in standard space
U = T_Nataf.X2U(X);   % [U, Jac] = X2U(Nataf,X,'Jac');

% samples in physical space
X = T_Nataf.U2X(U);

% marginal pdfs in physical space
n   = length(M);
xx  = cell(1,n);
f_X = cell(1,n);
for i = 1:n
   xx{i}  = X(i,:);
   f_X{i} = T_Nataf.Marginals(i).pdf(xx{i});
end

% marginal cdfs in physical space
n   = length(M);
F_X = cell(1,n);
for i = 1:n
   F_X{i} = T_Nataf.Marginals(i).cdf(xx{i});
end

% joint PDF and CDF (see Documentation for a plot example)
jointf_X = @(X) T_Nataf.pdf(X);
jointF_X = @(X) T_Nataf.cdf(X);

%% graphics
% plot the marginal PDFs
hl = cell(n,1);    % store the legends
figure;
for i = 1:n
   plot(xx{i},f_X{i},'.','LineWidth',2); hold on;
   hl{i} = sprintf('$f_{X%g}$',i);
end
xlabel('$X$','Interpreter','Latex','FontSize', 18);
ylabel('PDF','Interpreter','Latex','FontSize', 18);
hl = legend(hl,'Location','Best'); set(hl,'Interpreter','latex'); 
set(gca,'FontSize',15);

% plot the marginal CDFs
hl = cell(n,1);    % store the legends
figure;
for i = 1:n
   plot(xx{i},F_X{i},'.','LineWidth',2); hold on; 
   hl{i} = sprintf('$F_{X%g}$',i);
end
xlabel('$X$','Interpreter','Latex','FontSize', 18);
ylabel('CDF','Interpreter','Latex','FontSize', 18);
hl = legend(hl,'Location','Best'); set(hl,'Interpreter','latex');
set(gca,'FontSize',15);

% plot samples in physical and standard spaces
switch n;
   case 2;   % 2D
      figure;
      subplot(121); plot(X(1,:),X(2,:),'b.'); 
      title('Physical space','Interpreter','Latex','FontSize', 18);
      xlabel('$X_1$','Interpreter','Latex','FontSize', 18);
      ylabel('$X_2$','Interpreter','Latex','FontSize', 18);
      set(gca,'FontSize',15);
      subplot(122); plot(U(1,:),U(2,:),'r.'); axis equal; 
      title('Standard space','Interpreter','Latex','FontSize', 18);
      xlabel('$U_1$','Interpreter','Latex','FontSize', 18);
      ylabel('$U_2$','Interpreter','Latex','FontSize', 18);
      set(gca,'FontSize',15);
   case 3;   % 3D
      figure;
      subplot(121); plot3(X(1,:),X(2,:),X(3,:),'b.'); 
      title('Physical space','Interpreter','Latex','FontSize', 18);
      xlabel('$X_1$','Interpreter','Latex','FontSize', 18);
      ylabel('$X_2$','Interpreter','Latex','FontSize', 18);
      zlabel('$X_3$','Interpreter','Latex','FontSize', 18);
      set(gca,'FontSize',15);
      subplot(122); plot3(U(1,:),U(2,:),U(3,:),'r.'); axis equal; 
      title('Standard space','Interpreter','Latex','FontSize', 18);
      xlabel('$U_1$','Interpreter','Latex','FontSize', 18);
      ylabel('$U_2$','Interpreter','Latex','FontSize', 18);
      zlabel('$U_3$','Interpreter','Latex','FontSize', 18);
      set(gca,'FontSize',15);
   otherwise;
      disp('Non-supported plot');
end
%%END