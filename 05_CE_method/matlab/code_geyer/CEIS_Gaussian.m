function result = CEIS_Gaussian(input)

% shorthand
rho=input.rho;
max_iter=input.max_iter;
dim=input.dim;
N=input.N;
cykles=input.cykles;
f=input.f;
version=input.version;

% Vectors for simulation results
Pr=zeros(cykles,1);
l=zeros(cykles,1);
N_tot=zeros(cykles,1);
k_fin=zeros(cykles,1);

% Definition of parameters of the random variables (uncorrelated standard normal)
mu_init=zeros(1,dim);
Si_init=eye(dim);
Pi_init=1;

gamma_hat = zeros(max_iter+1,cykles);

figure('Position', [100, 100,1000,1000]);
hold on
%gp=@(x,y)0.1*(x-y).^2-(x+y)./sqrt(2)+2.5;



for i=1:cykles
    
    %% Initializing parameters
    
    gamma_hat(1,i) = 1;
    mu_hat=mu_init;
    Si_hat=Si_init;
    Pi_hat=Pi_init;
    
    %% Iteration
    
    for j=1:max_iter
        
        % Generate samples
        X=GM_sample(mu_hat,Si_hat,Pi_hat,N);
        
        % Count generated samples
        N_tot(i)=N_tot(i)+N;
        
        % Evaluation of the limit state function
        g=f(X);
        
        % Calculating h for the likelihood ratio
        h=h_calc(X,mu_hat,Si_hat,Pi_hat);
        
        % Check convergence
        if gamma_hat(j,i)==0
            if version=='CEGM'
                k_fin(i)=k;
            else
                k_fin=1;
            end
            break
        end
        
        % obtaining estimator gamma
        gamma_hat(j+1,i) = max(0,prctile(g,rho*100));
        
%         if gamma_hat(j+1,i)>0
%             fcontour(gp,[-2 4],'LevelList',gamma_hat(j+1,i),'LineColor','b','MeshDensity',100);
%         else
%              fcontour(gp,[-2 4],'LevelList',gamma_hat(j+1,i),'LineColor','r','MeshDensity',100);   
%         end
        
        % Indicator function
        I=(g<=gamma_hat(j+1,i));
        
        % Likelihood ratio
        W=mvnpdf(X,zeros(1,dim),eye(dim))./h;
%         
%         % plot samples
%         if mod(j,2)==0
%             plot(X(:,1),X(:,2),'o','Color','[0.4 0 0.9]')  
%         else
%             plot(X(:,1),X(:,2),'o','Color','[0 0.75 0]')
%         end
       
        % Parameter update
        
        if version == 'CEGM'
            
            % Define sample lsf values percentile to be used for DBSCAN
%             if j==1
%                 bdl=prctile(g,5);
%                 bdu=prctile(g,25);
%             else
%                 bdl=prctile(g,0);
%                 bdu=prctile(g,20);
%             end
%             
%             idl=(g<=bdu);
%             W2=W(idl);
%             g2=g(idl);
%             Y=X(idl,:);
%             idu=find(g2>bdl);
%             W2=W2(idu);
%             Y=Y(idu,:);
            
            % Cluster centers via DBSCAN algorithm
            %init.centers=centers_DBSCAN(Y,W2',version);
            %init.type='DBSCAN';
            
            init.nGM=1;
            %init.type='RAND';
            init.type='KMEANS';
            
%             figure
%             plot(init.centers(:,1),init.centers(:,2),'o')
%             hold on
%             plot(X(I,1),X(I,2),'*')
%             hold off
%             close all           
            
            % EM algorithm
            %k=numel(unique(IDX2));
            %dist = EMGM_soft_GM(X(I,:)',W(I),k);
            %dist = EMGM_soft_new(X(I,:)',W(I));
            dist = EMGM(X(I,:)',1./W(I),init);

            % Assigning the variables with updated parameters
            mu_hat=dist.mu';
            Si_hat=dist.si;
            Pi_hat=dist.pi';
            k=length(dist.pi);
            
        elseif version == 'CESG'
            
            % Closed-form update
            
            mu_hat=(W(I)'*X(I,:))./sum(W(I));
            Xo=bsxfun(@times,X(I,:)-mu_hat,sqrt(W(I)));
            Si_hat=Xo'*Xo/sum(W(I))+1e-6*eye(dim);
            
        else 
            error('Not a valid method')
        end
    end
    
    % store the needed steps
    l(i)=j;
    
    %% Calculation of Probability of failure
    
    W_final=mvnpdf(X,zeros(1,dim),eye(dim))./h;
    I_final=(g<=0);
    Pr(i)=1/N*sum(I_final.*W_final);
    
%     % plot final samples and limit state surface
% %     gp=@(x,y)min(0.1*(x-y).^2-(x+y)./sqrt(2)+2.5,object.beta+1/sqrt(object.dim)*(x+y))
%      gp=@(x,y)input.beta-1/sqrt(input.dim)*(x+y);
% %     gp=@(x,y)min([0.1.*(x-y).^2-(x+y)./sqrt(2)+3,0.1.*(x-y).^2+(x+y)./sqrt(2)+3,x-y+7./sqrt(2),y-x+7./sqrt(2)],[],2);
% %     gp=@(x,y)5-y-0.5.*(x-0.1).^2;
%      %figure('Position', [100, 100,1000,1000]);
%      j
%       plot(X(I_final,1),X(I_final,2),'*','Color','k')
%       I_nf=logical(1-I_final);
%       plot(X(I_nf,1),X(I_nf,2),'o','Color','k')
%       fcontour(gp,[-6 6],'LevelList',0,'LineColor','k','MeshDensity',1000);
%       box on
%       set(gca, 'FontSize', 16)
   
    
    %% Output of progress
    if mod(i,cykles/100)==0
        sprintf('%d%% completed',i/cykles*100)
    end
    
end




if version=='CESG'
    result.Version= 'CE method with single Gaussian';
elseif version=='CEGM'
    result.Version ='CE method with Gaussian mixture';
else
    error('Not a valid method')
end
result.Pr=Pr;
result.E_Pr=mean(Pr);
result.CoV_Pr=std(Pr)/result.E_Pr;
result.l=l;
result.E_l=mean(l);
result.N_tot=N_tot;
result.E_N_tot=mean(N_tot);
result.gamma_hat=gamma_hat;
if version=='CEGM'
    result.k_fin=k_fin;
    result.E_k_fin=mean(k_fin);
end


function X=GM_sample(mu,Si,Pi,N)

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


function h=h_calc(X,mu,Si,Pi)
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


