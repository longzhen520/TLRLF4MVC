function [pred_label]=TLRLF4MVC_Function(paras,transform)


X=paras.X;
gamma = paras.gamma;
lambda= paras.lambda;

cls_num = paras.M;
n_cluster = paras.N;
mu = paras.mu;
gt=paras.Y;
isCont = true; 


%------------Initializing parameters--------------
MaxIter =100; 
innerMax = 10;
display=1;
epson =1e-4;

BL=64;%binary length

V = size(X,2);
N = size(X{1},2);
% Random initialization
rng('default')
sumB=0;
for v=1:V
    rng(1);
    B{v}=randn(BL,N);
    C{v}=randn(BL,N);
    Y{v}=randn(BL,N);

    Q1{v}=zeros(BL,N);
    Q2{v}=zeros(BL,N);
    Q3{v}=zeros(BL,N);
    sumB=sumB+B{v};
end

U = cell(1,V);

B_hat=sumB/V;
rng(500)
D = B_hat(:,randsample(N, n_cluster));
[~,indx] = max( B_hat'*D,[],2);
G = sparse(indx,1:N,1,n_cluster,N,N);



XXT = cell(1,V);
for v= 1:V
    XXT{v} = X{v}*X{v}';
end

% clear HamDist ind initInd n_randm pcaW sel_sample view
%------------End Initialization--------------
beta1 = 1e-3; max_beta1 =1e5; 
beta2 = 1e-3; max_beta2 =1e5; 
beta3 = 1e-4; max_beta3 =mu;
pho =1.5;


%   figure;
% fprintf('\n Iteration: \t');
for iter = 1:MaxIter
%     fprintf('%d-th\t',iter);
    B_pre=B;
    B_hat_pre=B_hat;
%    fprintf('processing iter %d\n', iter);
 %---------Update U_v--------------
    for v = 1:V
        U{v} = B{v}*X{v}'/(XXT{v}+lambda*eye(size(X{v},1)));
        UX{v} = U{v}*X{v}; 
    end 
           

  %---------Update B_v--------------
    for v=1:V
         B{v}=(beta1*B_hat-Q1{v}+beta2*C{v}-Q2{v}+beta3*Y{v}-Q3{v}+UX{v})/(beta1+beta2+beta3+1);
         temp_B{v}=B{v}+Q1{v}/beta1;
         temp_C{v}=B{v}+Q2{v}/beta2;
         temp_Y{v}=B{v}+Q3{v}/beta3;
    end

  %---------Update B_hat=B-------------- pair-wise correlations

        B_hat = 0;
    for v=1:V
        B_hat = B_hat + 1/V*temp_B{v};
    end


 %---------Update C--------------%% high-order correlations

      C_tensor = cat(3, temp_C{:,:});   
     [temp_c]=prox_tnn(shiftdim(C_tensor, 2),gamma/beta2);
     c=shiftdim(temp_c,1);
          for v=1:V
        C{v} =c(:,:,v);  
          end
     Y_tensor = cat(3, temp_Y{:,:}); 

   %---------Update Y--------------%% low-frequency
%      [temp_y]=update_Y(shiftdim(Y_tensor, 2), transform, n_cluster );
%       y=shiftdim(temp_y,1);%% low-rank
%           for v=1:V
%         Y{v} =y(:,:,v);  
%           end
%% fast calculate
   Fin=transform.L(end-n_cluster+1:end,:); %%The eigenvectors corresponding to the top L largest eigenvalues 
    for v=1:V
         Y{v}= temp_Y{v}*Fin'*Fin;
    end
   

 %---------Update Q--------------%% 
        for v=1:V
        Q1{v}=Q1{v}+1.5*beta1*(B{v}-B_hat);  
        Q2{v}=Q2{v}+1.5*beta2*(B{v}-C{v});
        Q3{v}=Q3{v}+1.5*beta3*(B{v}-Y{v});  
        end
% 

  for v=1:V
    relChgB(v) = norm(B{v}- B_pre{v},'fro');
    relChgY(v) = norm(B{v}- Y{v},'fro');
    relChgC(v) = norm(B{v}- C{v},'fro');
    relChgBhat(v) = norm(B{v}- B_hat,'fro');
  end
   
    relChgBPath(iter) = max(relChgB);
    relChgCPath(iter) = max(relChgC);
    relChgYPath(iter) = max(relChgY);
    relChgBhatPath(iter) = max(relChgBhat);
    
    if (iter> 100) ||  ( relChgBPath(iter) < epson ) 
          disp(' !!!stopped by termination rule!!! ');  break;
    end

     %- continuation
      if  isCont
       nr1=relChgBhatPath(iter);
       nr2=relChgCPath(iter);
       nr3=relChgYPath(iter);  
             if iter >1 && nr1 > 0.95* nr1_pre
                beta1 = min(beta1*pho,max_beta1); 
             end
              if iter >1 && nr2 > 0.95* nr2_pre
                beta2 = min(beta2*pho,max_beta2); 
              end
               if iter >1 && nr3 > 0.95* nr3_pre
                beta3 = min(beta3*pho,max_beta3); 
               end
            nr1_pre =nr1;  nr2_pre =nr2;  nr3_pre =nr3;   
      end    
         

       
end


    %---------Update D and G--------------
 for iterInner = 1:innerMax
       D =fillmissing( normalize(B_hat*G'),'constant',1);       
        rho = .001; mu = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B_hat*G';
            D    = fillmissing(normalize(D-1/mu*grad),'constant',1);            
        end
        
       
        [~,indx] = max( B_hat'*D,[],2);
        G = sparse(indx,1:N,1,n_cluster,N,N);
    end
[~,pred_label] = max(G,[],1);

end



function [X] = prox_tnn(Y,rho)

[n1,n2,n3] = size(Y);
max12 = max(n1,n2);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
tnn = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
he(1)=sum(S);
S = max(S-rho,0);
tol = max12*eps(max(S));
r = sum(S > tol);
S = S(1:r);
X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';

% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    he(i)=sum(S);
    S = max(S-rho,0);
    tol = max12*eps(max(S));
    r = sum(S > tol);
    S = S(1:r);
    X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';    
    X(:,:,n3+2-i) = conj(X(:,:,i));
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
end
X = ifft(X,[],3);
end


% 
function [Y] = update_Y(M,transform,L)
 F=transform.L(end-L+1:end,:); %%The eigenvectors corresponding to the top L largest eigenvalues  
 M_hat= tmprod(M,F,3);%% low-frequency transform
 Y= tmprod(M_hat,F',3);%% inverse transform
end
