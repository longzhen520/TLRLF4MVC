function [pred_label, G] = LRLF4TMVC_Function(paras)
% Function to perform tensor low-rank and low-frequency for scalable multi-view clustering (TLRLF4MVC).
% Inputs:
%   - paras: structure containing parameters for the clustering
% Outputs:
%   - pred_label: predicted labels for each data point
%   - G: cluster indicator matrix


%% ------------IExtract parameters--------------
X=paras.X; %% anchor graph
lambda = paras.lambda; %% tune U
gamma= paras.gamma; %% consistency weight
L= paras.L;%% low-frequency parameter

cls_num = paras.M;
n_cluster = paras.N;
BL=32;%the size of embedding feature


%% ------------Initializing parameters--------------
innerMax = 10;
epson =1e-3;

display=1;

mu = 5e-4; max_mu =1e10 ; pho_mu =3;
beta=1e-3; max_beta =1.5; pho_beta =3;

V = size(X,2);
N = size(X{1},2);


% Random initialization
rng('default')
sumB=0;
for v=1:V
    rng(1);
    B{v}=randn(BL,N);
    Co{v}=zeros(BL,N);
    S{v}=zeros(BL,N);
    Lambda1{v}=zeros(BL,N);
    Lambda2{v}=zeros(BL,N);
    Y{v}=zeros(BL,N);
    sumB=sumB+B{v};
end

U = cell(1,V);

B_hat=sumB/V;
rng(500)
C = B_hat(:,randsample(N, n_cluster));
[~,indx] = max( B_hat'*C,[],2);
G = sparse(indx,1:N,1,n_cluster,N,N);



XXT = cell(1,V);
for v= 1:V
    XXT{v} = X{v}*X{v}';
end
% ------------End Initialization--------------

%% main fuction
Isconverg = 0;
iter = 0;
while(Isconverg == 0)
      iter = iter + 1;
    B_pre=B;Co_pre=Co;Y_pre=Y;
  
   fprintf('processing iter %d\n', iter);
 %---------Update U_v--------------
    for v = 1:V
        U{v} = B{v}*X{v}'/(XXT{v}+lambda*eye(size(X{v},1)));
        UX{v} = U{v}*X{v}; 
    end          
    
 %---------Update B_v--------------
    for v=1:V
         B{v}=normalize((mu*Co{v}-Lambda1{v}+beta*Y{v}-Lambda2{v}+UX{v})/(mu+beta+1));
         temp_C{v}=B{v}+Lambda1{v}/mu;
         temp_Y{v}=B{v}+Lambda2{v}/beta;
    end
      C_tensor = cat(3, temp_C{:,:}); 
 %---------Update C--------------%% low-rank component
     [temp_c,norm_TNN]=Prox_TNN(shiftdim(C_tensor, 2),gamma/mu);
     c=shiftdim(temp_c,1);%% low-rank
          for v=1:V
        Co{v} =c(:,:,v);  
          end
          
 %---------Update Y--------------%% low-frequency component    
      B_tensor = cat(3, temp_Y{:,:}); 
      b1=shiftdim(B_tensor, 2);
      y=shiftdim(Prox_TLFA(b1,0,[],L),1);%% low-frequency
        for v=1:V
        Y{v} =y(:,:,v);  
        end
        
 %---------Update Lambda--------------%% low-frequency component 
        for v=1:V
        Lambda1{v}=Lambda1{v}+mu*(B{v}-Co{v});  
        Lambda2{v}=Lambda2{v}+beta*(B{v}-Y{v});  
        end
      

  for v=1:V
    relChgB(v) = norm(B{v}- B_pre{v},'fro')/norm(B_pre{v});
  end
   
    relChgBPath(iter) = max(relChgB);
	if  display
        fprintf('iter: %4d, \t relChgX0:%4.4e\n', iter,  relChgBPath(iter));
    end
    
    if (iter> 20) ||  ( relChgBPath(iter) < epson ) 
          disp(' !!!stopped by termination rule!!! ');  break;
    end
         mu = min(mu*pho_mu, max_mu);  
         beta = min(beta*pho_beta, max_beta);    
end

B_hat = 0;
for v=1:V
    B_hat = B_hat + 1/V*B{v};
end
    %---------Update C and G--------------
 for iterInner = 1:innerMax
       C =fillmissing( normalize(B_hat*G'),'constant',1);       
        rho = .001; mu = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B_hat*G';
            C    = fillmissing(normalize(C-1/mu*grad),'constant',1);            
        end
        
       
        [~,indx] = max( B_hat'*C,[],2);
        G = sparse(indx,1:N,1,n_cluster,N,N);
    end
[~,pred_label] = max(G,[],1);

end


function [X, tnn] = Prox_TLFA(Y,rho,transform,rank_low)
[n1,n2,n3] = size(Y);
r3=min(rank_low,n3);
%disp(sprintf('n1: %d, n2: %d, n3: %d, r3: %d', n1, n2, n3,r3))
if nargin == 3
    if transform.l < 0
        error("property L'*L=L*L'=l*I does not holds for some l>0.");
    end
else    
    % fft is the default transform
    transform.L = @fft; transform.l = n3; transform.inverseL = @ifft;
end
X = zeros(n1,n2,n3);
tnn = 0;
    % efficient computing for fft transform
    Y = fft(Y,[],3);    
   X(:,:,1)= Y(:,:,1);
    %tnn = tnn+sum(S);
    % i=2,...,halfn3
    halfr3 = r3;
    for i = 2 : halfr3
        X(:,:,i)= Y(:,:,i);
        X(:,:,n3+2-i) = conj(X(:,:,i));
    end
    X = ifft(X,[],3);

end

function [X, tnn] = Prox_TNN(Y,rho)
[n1,n2,n3] = size(Y);
max12 = max(n1,n2);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
tnn = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
S = max(S-rho,0);
tol = max12*eps(max(S));
r = sum(S > tol);
S = S(1:r);
X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
tnn = tnn+sum(S);
trank = max(trank,r);

% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    S = max(S-rho,0);
    tol = max12*eps(max(S));
    r = sum(S > tol);
    S = S(1:r);
    X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';    
    X(:,:,n3+2-i) = conj(X(:,:,i));
     tnn = tnn+sum(S)*2;
%     trank = max(trank,r);
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    S = max(S-rho,0);
    tol = max12*eps(max(S));
    r = sum(S > tol);
    S = S(1:r);
    X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
%     tnn = tnn+sum(S);
%     trank = max(trank,r);
end
 tnn = tnn/n3;
X = ifft(X,[],3);
end