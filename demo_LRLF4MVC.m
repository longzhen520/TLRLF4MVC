clear all; 
close all; 
clc;
warning off;

% Add paths for datasets and utility functions
addpath('datasets', 'Utility')

% Define the list of datasets
ds = {'CCV', 'Caltech102', 'NUSWIDEOBJ', 'AwAfea', 'cifar10', 'YoutubeFace_sel'};

% Iterate through each dataset
for dsi = 1:1:length(ds)
    dataName = ds{dsi};   
    fprintf('\n Dataset:%s \n',dataName);
    data = dataName;


    % Load the corresponding dataset and set parameter
        switch data
            case 'CCV'      
                 load('CCV.mat');
                    %% low-rank+low-frequency
                    Lambda=0.00001;
                    Gamma=1;
                    L=30;
        
            case 'Caltech102'      
                 load('Caltech102.mat'); 
                    %% low-rank+low-frequency
                    Lambda=0.00001;
                    Gamma=1;
                    L=45; 
        
            case 'NUSWIDEOBJ'   
                 load('NUSWIDEOBJ.mat');   
                   %% low-rank+low-frequency   
                     Lambda=0.00001;
                     Gamma=500;
                     L=40; 
        
            case 'AwAfea'      
                load('AwAfea.mat'); 
                  %% low-rank+low-frequency   
                     Lambda=0.00001;
                     Gamma=100;
                     L=60;
        
            case 'cifar10'   
                 load('cifar10.mat'); 
                      %% low-rank+low-frequency   
                     Lambda=1e-5;
                     Gamma=10;
                     L=15;
           
            case 'YoutubeFace_sel'      
                 load('YoutubeFace_sel.mat');  
                  %% low-rank+low-frequency   
                     Lambda=0.00001;
                     Gamma=300;
                     L=40;  
         
end

resultsAll = [];
V = size(X,2);
N= length(Y);

fprintf('The Nonlinear Anchor Embeeding：');
for it = 1:V
fprintf('%d \t',it);
    [~,Anchor{it}] = AnchorGEN(X{it},9,20,1);
    dist = EuDist2(X{it},Anchor{it},0); 
    sigma = mean(min(dist,[],2).^0.5)*2;
    feaVec = exp(-dist/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feaVec', mean(feaVec',2));
end

clear feaVec dist sigma dist Anchor it

    % Iterate through different parameter values
    for l=1:length(L)    
        for i=1:length(Lambda)
                for s=1:length(Gamma)
                    %% parameter setting
                    cls_num = length(unique(Y));
                    n_cluster = numel(unique(Y));
                    V = length(X); 
                    N = size(X{1},2); 
                    paras.X=X;
                    paras.lambda=Lambda(i);
                    paras.gamma=Gamma(s);
                    paras.L=L(l);
                    paras.M=cls_num;
                    paras.N=n_cluster;

% ---------------------------------CLUSTERING------------------------------------------------
        tic;
        [pred_label,G]= LRLF4TMVC_Function(paras);
        execution_times= toc;
% -------------------------------------- ----------------------------------------------------
     
        % Evaluate clustering results
        res_cluster = Clustering8Measure(Y, pred_label);
        fprintf(['\tACC:%.4f\t NMI:%.4f\t Purity:%.4f\t F-score:%.4f\t PRE:%.4f\t REC:%.4f\t AR:%.4f\t Entropy:%.4f\t ,Times = %.2f\n '],res_cluster,execution_times);
      
            end
        end
    end
end
    