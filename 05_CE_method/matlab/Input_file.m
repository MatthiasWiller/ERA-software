clear all
close all
clc

format long

%% Predefinitions

% Definition of additional values
object.rho=0.1; % Sample Quantile or 1/nchain in SIS approach
object.max_iter=100; % Maximum number of iteration steps per simulation
object.dim =1; % Dimensions of the problem
object.N=1e3; % Definition of number of samples per level
object.cykles=1; % # simulation runs

% CEGM for CE method with Gaussian mixture 
% CESG for CE method with single Gaussian 
% SISGM for sequential importance sampling with GM 
% CEvMFNM for adaptive vMFN mixture
% new_method for new method (Combination of SIS with CE method)

object.version='CEGM'; 

%% Definition of limit state function

% Onedimensional lsf
%object.f=@(x)5+sqrt(2).*x;

% Convex limit state function
object.f=@(x)0.1*(x(:,1)-x(:,2)).^2-(x(:,1)+x(:,2))./sqrt(2)+2.5;

% Parabolic/concave limit state function
%object.f=@(x)5-x(:,2)-0.5.*(x(:,1)-0.1).^2;
%object.f=@(x)5-x(:,2)-0.5.*(x(:,1)-0.1).^2+1e-3*sum(sin(100*x),2); % Additional noisy term

% Series system problem
%object.f=@(x)min([0.1.*(x(:,1)-x(:,2)).^2-(x(:,1)+x(:,2))./sqrt(2)+3,0.1.*(x(:,1)-x(:,2)).^2+(x(:,1)+x(:,2))./sqrt(2)+3,x(:,1)-x(:,2)+7./sqrt(2),x(:,2)-x(:,1)+7./sqrt(2)],[],2);

% Parallel system problem
%object.f=@(x)max([x(:,1).^2-5*x(:,1)-8*x(:,2)+6,-16*x(:,1)+x(:,2).^2+40],[],2);

% Linear limit state function with single failure domain
%object.beta=3.5; % reliability index
%object.f=@(x)object.beta-1/sqrt(object.dim)*sum(x,2);

% Linear limit state function with two failure domains
%object.beta=3.5; % reliability index
%object.f=@(x)min(object.beta*sqrt(object.dim)+sum(x,2),object.beta*sqrt(object.dim)-sum(x,2));

% Convex limit state function combined with linear limit state function
%object.beta=3.2; % reliability index
%object.f=@(x)min(0.1*(x(:,1)-x(:,2)).^2-(x(:,1)+x(:,2))./sqrt(2)+2.5,object.beta+1/sqrt(object.dim)*sum(x,2));

% Exponential limit state function with one failure domain
%object.Ca=10; % parameter for exponential limit state functions
%object.f=@(x)object.Ca+sum(log(normcdf(-x)),2);

% Exponential limit state function with two failure domains
%object.Ca=10; % parameter for exponential limit state functions
%object.Cb=0.03; % additional parameter for exp. limit state function with two design points
%object.f=@(x)min([object.Ca+sum(log(normcdf(-x)),2),-object.Cb-sum(log(normcdf(-x)),2)],[],2);


%% Simulation

if strcmp('CEGM',object.version) || strcmp('CESG',object.version)
    
    result=CEIS_Gaussian(object)
    
elseif strcmp('SISGM',object.version)
    
    result=SIS_GM(object)

elseif strcmp('CEvMFNM',object.version)
    
    result=CEIS_vMFNM(object)
    
elseif strcmp('new_method',object.version)
    
    result=CEIS_new_method(object)    
    
else
    
    error('Choose available option')

end
