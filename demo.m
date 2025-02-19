clear all; 
close all; 
clc;
warning off;

% Add paths for datasets and utility functions
addpath('datasets','transform', 'Utility')


% Define the list of datasets
ds = {'CCV','ALOI' };

% Iterate through each dataset
for dsi = 1:1:length(ds)
    dataName = ds{dsi};   
    fprintf('\n Dataset:%s \n',dataName);
    data = dataName;

     
    % Load the corresponding dataset and set parameter

     switch data
            case 'CCV'      
               load('CCV.mat');  load('CCV_transform.mat')
                    Lambda=1e-2;
                    Gamma=1e-3;
                    Mu=1e-1;
        
            case 'Caltech102'      
               load('Caltech102.mat'); load('Caltech102_transform.mat')
                    Lambda=1e-4;
                    Gamma=1e-1;
                    Mu=1e-4; 

            case 'ALOI'      
                load('ALOI.mat');  load('ALOI_transform.mat')
                     Lambda=1;
                     Gamma=1e-3;
                     Mu=1e-1;   

            case 'NUSWIDEOBJ'   
                   load('NUSWIDEOBJ.mat');load('Nus_transform.mat')
                     Lambda=1e-4;
                     Gamma=1;
                     Mu=1e-4; 
        
            case 'AwAfea'      
                 load('AwAfea.mat'); load('AWA_transform.mat');
                     Lambda=1e-4;
                     Gamma=1e-1;
                     Mu=1e-1;
        
            case 'cifar10'   
                 load('cifar10.mat');  load('cifar_transform.mat')
                     Lambda=1e-4;
                     Gamma=1e-3;
                     Mu=0.1;
        
        end


V = size(X,2);
N= length(Y);


fprintf('The Nonlinear Anchor Embeeding：');
for it = 1:V
fprintf('%d \t',it);
     [~,Anchor{it}] = AnchorGEN(X{it},10,20,1);
    dist = EuDist2(X{it},Anchor{it},0); 
    sigma = mean(min(dist,[],2).^0.5)*2;
    feaVec = exp(-dist/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));
end


clear feaVec dist sigma dist Anchor it


% Gamma=[1e-4,1e-3,1e-2,1e-1,1];
% Lambda=[1e-4,1e-3,1e-2,1e-1,1];
% Mu=[1e-4,1e-3,1e-2,1e-1,1];
%% generate GFT
%     for v=1:V
%         X{v}=normalize(X{v},2);
%     end
%       pred_label = litekmeans(X_pre,cls_num,'MaxIter',1000, 'Replicates',3);
%       options = [];
%       options.NeighborMode = 'Supervised';
%       options.gnd = pred_label;
%       options.WeightMode = 'Binary';
%       options.t = 1;
%       W = constructW(X_pre,options);
%       [Fin,S]=eig(full(W));


transform.L =Fin; 

% K=[2:1:10];
 total_iterations =  length(Gamma) * length(Lambda)* length(Mu);
 current_iteration = 0;
    % Iterate through different parameter values
  
        for i=1:length(Lambda)
                for s=1:length(Gamma)
                     for k=1:length(Mu)
                    %% parameter setting
                    cls_num = length(unique(Y));
                    n_cluster = numel(unique(Y));
                    V = length(X); 
                    N = size(X{1},2); 
                    paras.X=X;
                    paras.lambda=Lambda(i);
                    paras.gamma=Gamma(s);
                     paras.mu=Mu(k);
                    paras.Y=Y;
%                     paras.k=K(k);
                    paras.M=cls_num;
                    paras.N=n_cluster;
% ---------------------------------CLUSTERING------------------------------------------------
       t=tic;
        [pred_label]= TLRLF4MVC_Function(paras,transform);
       t2=toc(t);
% -------------------------------------- ----------------------------------------------------
       Result= Clustering8Measure(Y, pred_label);
      fprintf('================== Result =====================\n');
     fprintf('    %5.6s \t   %5.6s \t  %5.6s  \t   %5.6s  \t   %5.6s \t   %5.6s \t   %5.6s \t   %5.6s \t   %5.6s \n','Time','ACC', 'NMI','Purity', 'Fscore','Precision','Recall','ARI','Entropy');

    fprintf('%5.4f \t %5.4f \t   %5.4f \t    %5.4f \t %5.4f \t %5.4f \t   %5.4f  \t %5.4f \t   %5.4f  \n',t2,Result);

% -------------------------------------- ----------------------------------------------------


        end
                end
        end

    end
    
