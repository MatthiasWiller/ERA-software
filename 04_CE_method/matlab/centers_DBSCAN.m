function C=centers_DBSCAN(X,W,version)
% Identify the number of clusters via DBSCAN algorithm and the
% corresponding cluster centers
%   X: N x d data matrix (Number of samples x dimensions)
%   W: N x 1 vector of sample weights

% scaling the weights
W_norm=W/mean(W);

d=size(X,2);

% Parameters for the DBSCAN algorithm
if strcmp('GM',version) % GM
    
    epsilon=mean(mean(squareform(pdist(X))))/(8/d); % radius
    MinW=sum(W_norm)/(size(W_norm,2)/(8/d)); % minimum scaled weight per cluster
    
elseif strcmp('SIS',version) % SIS
    
    
    epsilon=mean(mean(squareform(pdist(X))))/(8/d); % radius
    MinW=sum(W_norm)/(size(W_norm,2)/(4/d)); % minimum scaled weight per cluster

elseif strcmp('vMFNM',version) % vMFNM 
    
    % Cosine similarity
    
    epsilon=(std(pdist(X,'cosine')))^(2*d);
    %epsilon=mean(mean(squareform(pdist(X))))/(8/d) % radius
    MinW=1;
    %MinW=sum(W_norm)/(size(W_norm,2)/(8/d)) % minimum scaled weight per cluster
 
elseif strcmp('new_method',version) 
    
    % Cosine similarity
    epsilon=(std(pdist(X,'cosine')))^(2*d);
    %epsilon=mean(mean(squareform(pdist(X))))/(8/d) % radius
    MinW=1;
    %MinW=sum(W_norm)/(size(W_norm,2)/(8/d)) % minimum scaled weight per cluster    
    
else
    error('Not available option')
end


% DBSCAN algorithm
IDX=DBSCAN_weights(X,W_norm,epsilon,MinW);

nnz(IDX)

% Binary matrix of weighted likelihoods for all considered samples
R=dummyvar(IDX+1);
R=R(:,2:end);
wR=bsxfun(@times,R,W');

% Identify weighted cluster centers
C=bsxfun(@times,wR'*X,1./(sum(wR)'));

% Plot results
%PlotClusterinResult(X, IDX);
%title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinW = ' num2str(MinW) ')']);
%close
end


function IDX=DBSCAN_weights(X,W,epsilon,MinW)

%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

C=0;

n=size(X,1);
IDX=zeros(n,1);

%D=pdist2(X,X);
D=pdist2(X,X,'cosine');

visited=false(n,1);

for i=1:n
    if ~visited(i)
        visited(i)=true;
        
        Neighbors=RegionQuery(i);
        w_N=W(Neighbors);

        if sum(w_N)>=MinW 
            C=C+1;
            ExpandCluster(i,Neighbors,C);
            %fprintf('%d\n',numel(Neighbors));
        end
        
    end
    
end

    function ExpandCluster(i,Neighbors,C)
        IDX(i)=C;
        
        k = 1;
        while true
            j = Neighbors(k);
            
            if ~visited(j)
                visited(j)=true;
                Neighbors2=RegionQuery(j);
                w_N2=W(Neighbors2);
                if sum(w_N2)>=MinW %&& length(unique(Neighbors))>1
                    Neighbors=[Neighbors Neighbors2];   % ok
                end
            end
            if IDX(j)==0
                IDX(j)=C;
            end
            
            k = k + 1;
            if k > numel(Neighbors)
                break;
            end
        end
    end

    function Neighbors=RegionQuery(i)
        Neighbors=find(D(i,:)<=epsilon);
    end

end


