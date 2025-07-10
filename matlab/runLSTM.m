function [fitness, net, result] = runLSTM(params, filename)
% RUNLSTM Train and evaluate an LSTM network.
%   [FITNESS, NET, RESULT] = RUNLSTM(PARAMS, FILENAME) trains an LSTM
%   regression network using the provided PARAMS = [hiddenUnits, learnRate,
%   maxEpochs] on data stored in FILENAME. It returns FITNESS (test RMSE),
%   the trained network NET, and a structure RESULT with metrics.

%% 1. Basic parameters
numHiddenUnits = round(params(1));
learnRate = params(2);
maxEpochs = round(params(3));

%% 2. Load data
raw = xlsread(filename);
numSamples = numel(raw);
kim = 15;   % delay steps
zim = 1;    % predict zim time steps ahead

res = zeros(numSamples - kim - zim + 1, kim + 1);
for i = 1:numSamples - kim - zim + 1
    res(i,:) = [raw(i:i+kim-1).', raw(i+kim+zim-1)];
end

outdim = 1;
trainRatio = 0.7;
num_train = round(trainRatio * size(res,1));
features = size(res,2) - outdim;

P_train = res(1:num_train,1:features)';
T_train = res(1:num_train,features+1:end)';
P_test  = res(num_train+1:end,1:features)';
T_test  = res(num_train+1:end,features+1:end)';
M = size(P_train,2);
N = size(P_test,2);

%% 3. Normalization
[P_train, ps_input] = mapminmax(P_train,0,1);
P_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train,0,1);
t_test = mapminmax('apply', T_test, ps_output);

P_train = reshape(double(P_train),features,1,1,M);
P_test  = reshape(double(P_test),features,1,1,N);
t_train = t_train';
t_test  = t_test';

p_train = cell(M,1);
for i = 1:M
    p_train{i} = P_train(:,:,1,i);
end
p_test = cell(N,1);
for i = 1:N
    p_test{i} = P_test(:,:,1,i);
end

%% 4. Network architecture
layers = [
    sequenceInputLayer(features)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', learnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', round(maxEpochs*0.8), ...
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization', 1e-4, ...
    'ExecutionEnvironment', 'auto', ...
    'Verbose', false);

%% 5. Train network
net = trainNetwork(p_train, t_train, layers, options);

%% 6. Predict
predTrain = predict(net, p_train);
predTest = predict(net, p_test);

T_train_hat = mapminmax('reverse', predTrain, ps_output);
T_test_hat  = mapminmax('reverse', predTest, ps_output);

%% 7. Metrics
rmseTrain = sqrt(sum((T_train_hat' - T_train).^2) / M);
rmseTest  = sqrt(sum((T_test_hat' - T_test).^2) / N);

R2_train = 1 - norm(T_train - T_train_hat')^2 / norm(T_train - mean(T_train))^2;
R2_test  = 1 - norm(T_test - T_test_hat')^2 / norm(T_test - mean(T_test))^2;

maeTrain = sum(abs(T_train_hat' - T_train)) / M;
maeTest  = sum(abs(T_test_hat' - T_test)) / N;

mbeTrain = sum(T_train_hat' - T_train) / M;
mbeTest  = sum(T_test_hat' - T_test) / N;

mapeTrain = sum(abs((T_train_hat' - T_train) ./ T_train)) / M;
mapeTest  = sum(abs((T_test_hat' - T_test) ./ T_test)) / N;

fitness = rmseTest;

result = struct('trainRMSE', rmseTrain, 'testRMSE', rmseTest, ...
    'R2_train', R2_train, 'R2_test', R2_test, ...
    'MAE_train', maeTrain, 'MAE_test', maeTest, ...
    'MBE_train', mbeTrain, 'MBE_test', mbeTest, ...
    'MAPE_train', mapeTrain, 'MAPE_test', mapeTest);

end

