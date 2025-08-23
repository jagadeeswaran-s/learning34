clc; clear; close all;

%% Application: Concrete Property Prediction (MIMO)
disp('=== MIMO Assignment: Concrete Properties ===');

%% Load dataset
data = xlsread('Concrete_Data.xls');

% Choose inputs & outputs
X = data(:,1:6);              % Inputs: Cement, Slag, Fly Ash, Water, SP, Coarse Agg
T = data(:,7:8);              % Outputs: Fine Aggregate, Strength

% Output names for plotting
outputNames = {'Fine Aggregate','Strength'};

%% Normalize
Xn = (X - min(X)) ./ (max(X) - min(X));
Tn = (T - min(T)) ./ (max(T) - min(T));

%% Parameters
Nh   = 15;    % hidden neurons
Nit  = 500;   % epochs
eta  = 0.05;  % learning rate
trainRatio = 0.7;

% Train/test split
N = size(Xn,1);
idx = randperm(N);
Ntrain = round(trainRatio * N);
Xtrain = Xn(idx(1:Ntrain),:);
Ttrain = Tn(idx(1:Ntrain),:);
Xtest  = Xn(idx(Ntrain+1:end),:);
Ttest  = Tn(idx(Ntrain+1:end),:);

%% Train
[W1,b1,W2,b2,MSE] = train_mimo(Xtrain, Ttrain, Nh, Nit, eta);

%% Predict
Z1 = W1*Xn' + b1;
A1 = tansig(Z1);
Z2 = W2*A1 + b2;
Y_hat = purelin(Z2);

% Denormalize
Y_pred = Y_hat' .* (max(T)-min(T)) + min(T);

%% Errors
trainPred = (W2*tansig(W1*Xtrain' + b1) + b2)' .* (max(T)-min(T)) + min(T);
testPred  = (W2*tansig(W1*Xtest'  + b1) + b2)' .* (max(T)-min(T)) + min(T);
trainError = mean((Ttrain.*(max(T)-min(T))+min(T) - trainPred).^2,'all');
testError  = mean((Ttest.*(max(T)-min(T))+min(T) - testPred).^2,'all');

%% Report
disp(['Dataset Used        : Concrete Data']);
disp(['Hidden Neurons      : ',num2str(Nh)]);
disp(['Epochs              : ',num2str(Nit)]);
disp(['Learning Rate       : ',num2str(eta)]);
disp(['Final Training MSE  : ',num2str(MSE(end))]);
disp(['Training Error (denorm) : ',num2str(trainError)]);
disp(['Testing Error (denorm)  : ',num2str(testError)]);

%% Plot Actual vs Predicted for each output
No = size(T,2);
for k = 1:No
    figure;
    plot(T(:,k),'b-o','LineWidth',1.2); hold on;
    plot(Y_pred(:,k),'r-*','LineWidth',1.2);
    xlabel('Sample Index'); ylabel(outputNames{k});
    title(['MIMO: Actual vs Predicted ',outputNames{k}]);
    legend('Actual','Predicted'); grid on;
end

%% Combined subplot
figure;
for k = 1:No
    subplot(No,1,k);
    plot(T(:,k),'b-o','LineWidth',1.2); hold on;
    plot(Y_pred(:,k),'r-*','LineWidth',1.2);
    ylabel(outputNames{k});
    legend('Actual','Predicted'); grid on;
end
xlabel('Sample Index');
sgtitle('MIMO: Actual vs Predicted (All Variables)');

%% Plot MSE
figure;
plot(1:Nit, MSE,'k','LineWidth',1.5);
xlabel('Epochs'); ylabel('MSE');
title('MIMO: Training Error vs Epochs');
grid on;

disp('=== End of MIMO Assignment ===');
pause;
