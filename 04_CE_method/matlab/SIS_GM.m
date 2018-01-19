function result = SIS_GM(input)

% shorthand
dim = input.dim;
nsamlev=input.N;
nchain=input.rho*nsamlev;
lenchain=nsamlev/nchain;
gfun=input.f;
cykles=input.cykles;
max_iter=input.max_iter;


%%%%
% number of distribution in Gaussian mixture
%nGM = 2;
%%%%

% burn-in
burn = 0;

% tolerance of COV of weights
tolCOV = 1.5;

% initialize samples
uk=zeros(nsamlev,dim);

l_tot=zeros(cykles,1);


numsim = zeros(cykles,1);
k_fin=zeros(cykles,1);

for n=1:cykles

clear uk gk sigmak Sk wk wnork 

%%% Step 1
% Perform the first Monte Carlo simulation
for k=1:nsamlev 

   % Do the simulation (create array of random numbers)
   u = randn(dim,1);
   
   uk(k,:) = u;
   
   gk(k) = gfun(u');  

   numsim(n) = numsim(n)+1;
    
end


% set initial subset and failure level
m = 1;
gmu = mean(gk);
sigmak(m) = 50*gmu;

while 1 && m<max_iter

    %%% Step 2 and 3
    % compute sigma and weights
    if m == 1
        sigma2 = fminbnd(@(x)abs(std(normcdf(-gk/x))/mean(normcdf(-gk/x))-tolCOV),0,10.0*gmu);
        sigmak(m+1) = sigma2;
        wk = normcdf(-gk/sigmak(m+1));
    else     
        sigma2 = fminbnd(@(x)abs(std(normcdf(-gk/x)./normcdf(-gk/sigmak(m)))/mean(normcdf(-gk/x)./normcdf(-gk/sigmak(m)))-tolCOV),0,sigmak(m));
        sigmak(m+1) = sigma2;
        wk = normcdf(-gk/sigmak(m+1))./normcdf(-gk/sigmak(m));
    end
    
    %%% Step 4
    % compute estimate of expected w
    Sk(m) = mean(wk);
    
    % Exit algorithm if no convergence is achieved
    if Sk(m)==0    
        pfk=0;
        break
    end
    
    
    % plot samples
    %plot(uk(:,1),uk(:,2),'x')
    
    % compute normalized weights
    wnork = wk./Sk(m)/nsamlev;
    
    init.nGM=2;
    %init.type='RAND';
    init.type='KMEANS';
    
    % fit Gaussian Mixture
    %[~, dist] = emgm2(uk',wnork',nGM);
    dist = EMGM(uk',wnork',init);
    
    %%% Step 5
    % resample
    ind = randsample(nsamlev,nchain,true,wnork);
        
    % seeds for chains
    gk0 = gk(ind);
    uk0 = uk(ind,:);
    
    %%% Step 6
    % perform M-H
    
    counta = 0;
    count = 0;
    
    % initialize chain acceptance rate
    alphak = zeros(nchain,1);    
    
    % delete previous samples
    gk = [];
    uk = [];
       
    for k=1:nchain
        
        % set seed for chain
        u0 = uk0(k,:);
        g0 = gk0(k);
           
        
        for j=1:lenchain+burn                       
            
            count = count+1;
            
            if j == burn+1
               count = count-burn; 
            end
            
            % get candidate sample from conditional normal distribution 
            indw = randsample(length(dist.pi),1,true,dist.pi);
            ucand = mvnrnd(dist.mu(:,indw),dist.si(:,:,indw));
            %ucand = muf + (Acov*randn(dim,1))';
            
            % Evaluate limit-state function              
            gcand = gfun(ucand);             
            
            numsim(n) = numsim(n)+1;

            % compute acceptance probability
            pdfn = 0;
            pdfd = 0;
            for i = 1:length(dist.pi)
                pdfn = pdfn+dist.pi(i)*mvnpdf(u0',dist.mu(:,i),dist.si(:,:,i));
                
                pdfd = pdfd+dist.pi(i)*mvnpdf(ucand',dist.mu(:,i),dist.si(:,:,i));
                
            end
            
            alpha = min(1,normcdf(-gcand/sigmak(m+1))*prod(normpdf(ucand))*pdfn/normcdf(-g0/sigmak(m+1))/prod(normpdf(u0))/pdfd);
            
            alphak(k) = alphak(k)+alpha/(lenchain+burn);
          
            % check if sample is accepted
            uhelp = rand;
            if uhelp <= alpha

                uk(count,:) = ucand;
                gk(count) = gcand;                
                u0 = ucand;
                g0 = gcand;
                 
            else
                
                uk(count,:) = u0;
                gk(count) = g0;
                
            end
                                
        end
           
            
    end
    
    uk = uk(1:nsamlev,:);
    gk = gk(1:nsamlev);    
    
    % compute mean acceptance rate of all chains in level m
    accrate(m) = mean(alphak);
    COV_Sl = std((gk < 0)./normcdf(-gk/sigmak(m+1)))/mean((gk < 0)./normcdf(-gk/sigmak(m+1)))
    if COV_Sl < 0.01   
        break;
    end                
    
    m = m+1;
    
end
k_fin(n)=length(dist.pi);

const=prod(Sk);

% probability of failure
pf = mean((gk < 0)./normcdf(-gk/sigmak(m+1)))*const;

%for k=1:nsamlev
%   
%    dgk(k,:) = dgfun(uk(k,:));
%    
%end

%sigmadp = fminbnd(@(x)abs(max(std(repmat(-1/x*normpdf(gk'/x)./normcdf(-gk'/sigmak(m+1)),1,2).*dgk)./abs(mean(repmat(-1/x*normpdf(gk'/x)./normcdf(-gk'/sigmak(m+1)),1,2).*dgk)))-tolCOV),0,sigmak(m+1));

%dpf = mean(repmat(-1/sigmadp*normpdf(gk'/sigmadp)./normcdf(-gk'/sigmak(m+1)),1,2).*dgk)*const;

pfk(n)=pf;

%dpfk(n,:) = dpf;

accfin(n) = accrate(m);     

l_tot(n)=m+1;
n
end

result.Version='SIS';
result.Pr=pfk;
result.E_Pr=mean(pfk);
result.CoV_Pr=std(pfk)/result.E_Pr;
result.l=l_tot;
result.E_l=mean(l_tot);
result.N_tot=numsim;
result.E_N_tot=mean(numsim);
result.k_fin=k_fin;
result.E_k_fin=mean(k_fin);

end

