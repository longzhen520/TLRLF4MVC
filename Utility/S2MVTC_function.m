function pred_label=S2MVTC_function(paras)


X=paras.X;
beta = paras.beta;
lambda = paras.lambda;
L = paras.L;
cls_num = paras.M;
n_cluster = paras.N;

%------------Initializing parameters--------------
MaxIter = 7; 
innerMax = 10;

BL=64;%binary length
alpha=1;
viewNum = size(X,2);
N = size(X{1},2);
sX=[BL,N,viewNum];
rand('seed',100);
sumB=0;
for it=1:viewNum
sel_sample = X{it}(:,randsample(N, 1000),:);
[pcaW, ~] = eigs(cov(sel_sample'), BL);
B{it} = pcaW'*X{it};
Y_tensor{it}=B{it};
sumB=sumB+B{it};
end

B_hat=sumB/viewNum;

U = cell(1,viewNum);

rand('seed',500);
C = B_hat(:,randsample(N, n_cluster));
[~,indx] = max( B_hat'*C,[],2);
G = sparse(indx,1:N,1,n_cluster,N,N);


XXT = cell(1,viewNum);
for view = 1:viewNum
    XXT{view} = X{view}*X{view}';
end
clear HamDist ind initInd n_randm pcaW sel_sample view
%------------End Initialization--------------

fprintf('\n Iteration: \t');
for iter = 1:MaxIter
    fprintf('%d-th\t',iter);
    %---------Update Ui--------------
    for v = 1:viewNum
            sum_B{v}=zeros(size(B{v}));

        U{v} = B{v}*X{v}'/(XXT{v}+lambda*eye(size(X{v},1)));
        UX{v} = U{v}*X{v}; 
          for u=1:viewNum
             if u==v
                 sum_B{v}=sum_B{v};
             else
                 sum_B{v}=sum_B{v}+1/viewNum*B{u};
             end
          end
    end
    
           
    
    %---------Update B_v--------------
     temp_sum=0;
    for v=1:viewNum
         B{v}=normalize((beta*(B_hat)+UX{v}+alpha*Y_tensor{v})/(beta+2));
        temp_sum=temp_sum+1/viewNum*B{v};
    end
    
    %% update Y_tensor
    B_tensor = cat(3, B{:,:});   
    SumP=zeros(sX);
    for i=3:3
     b=shiftdim(B_tensor, i-1);
     p=shiftdim(prox_tnn(b,0,[],L),4-i);
     %SumP=SumP+w(i)*p;
    end
     tensor_Y = p;
      for v=1:viewNum
       Y_tensor{v} =tensor_Y(:,:,v);
      end
    %update B_hat
     B_hat = normalize(temp_sum);
    alpha=alpha*1.1;

    %%check convergence
      %%%% for v=1:viewNum
     %%%%      error(v)=norm(B{v}-B_pre{v},'fro')/norm(B{v},'fro');
   %%%% end
   %%%% RE(iter)=min(error);
   %%%%    disp(RE(iter))
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





function [X, tnn] = prox_tnn(Y,rho,transform,rank_low)

[n1,n2,n3] = size(Y);
r3=min(rank_low,n3);
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
if isequal(transform.L,@fft)
    Y = fft(Y,[],3);    
    X(:,:,1)= Y(:,:,1);
    halfr3 = r3;
    for i = 2 : halfr3
 X(:,:,i)= Y(:,:,i);
        X(:,:,n3+2-i) = conj(X(:,:,i));
       end
       X = ifft(X,[],3);
else
    % other transform
    Y = lineartransform(Y,transform);
    for i = 1 : r3
        [U,S,V] = svd(Y(:,:,i),'econ');
        S = diag(S);
        r = length(find(S>rho));
        if r >= 1
            S = S(1:r)-rho;
            X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
            tnn = tnn+sum(S);
        end
    end
    tnn = tnn/transform.l;
    X = inverselineartransform(X,transform);
end
end