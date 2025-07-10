function [rmse, predictions, targets] = runLSTM(params, filename)
% RUNLSTM Example LSTM training function.
%   [RMSE, PREDICTIONS, TARGETS] = RUNLSTM(PARAMS, FILENAME) trains a simple
%   LSTM regression network using data loaded from the Excel file FILENAME.
%   The Excel data is read using READMATRIX for compatibility with newer
%   MATLAB versions. Each column except the last is treated as an input
%   feature and the last column is the regression target.
%
%   PARAMS is a vector with three elements:
%       [hiddenUnits, learnRate, epochs]
%   representing the number of hidden units, learning rate and number of
%   training epochs respectively.
%
%   This is a minimal example used for optimization. In practice you may
%   need to adapt the preprocessing and network architecture to your data.
%
%   Example:
%       params = [50, 0.005, 100];
%       [rmse] = runLSTM(params, 'data.xlsx');
%
%   See also READMATRIX, TRAINNETWORK.

% Load dataset from Excel file
% Data should be organized with features in columns 1:end-1 and the target
% in the last column.
% Prior versions used XLSREAD, but READMATRIX is recommended starting in
% R2019b and later.
data = readmatrix(filename);
inputs = data(:, 1:end-1)';
targets = data(:, end)';

% Map parameters
numHiddenUnits = round(params(1));
learnRate      = params(2);
maxEpochs      = round(params(3));
inputSize      = size(inputs, 1);

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', maxEpochs, ...
    'GradientThreshold', 1, ...
    'Verbose', false);

net = trainNetwork(inputs, targets, layers, options);

predictions = predict(net, inputs);
rmse = sqrt(mean((predictions - targets).^2));
end
