function [best_params, best_rmse, curve] = acpso_lstm(filename, opts)
% ACPSO_LSTM Optimize LSTM hyperparameters using an adaptive PSO.
%   [BEST_PARAMS, BEST_RMSE, CURVE] = ACPSO_LSTM(FILENAME) runs the
%   adaptive particle swarm optimizer on data stored in FILENAME using the
%   external function runLSTM. It returns the best hyperparameters,
%   the associated RMSE, and the convergence curve.
%
%   opts is a structure with optional fields:
%       .nParticles  - number of particles (default 20)
%       .nIter       - maximum iterations (default 20)
%       .bounds      - 2xD matrix of [lb; ub] for each dimension
%
%   The optimizer searches for three parameters:
%       hidden units, learning rate, training epochs.
%
%   Example:
%       [p, rmse] = acpso_lstm('data.xlsx');
%
%   NOTE: This function expects a companion function RUNLSTM that accepts
%   the real-valued parameter vector and dataset filename, and returns the
%   RMSE on the validation or test set.

if nargin < 2
    opts = struct();
end

% Default options
nParticles = getOption(opts, 'nParticles', 20);
maxIter     = getOption(opts, 'nIter', 20);

% Parameter bounds: [hidden units, learning rate, epochs]
if isfield(opts, 'bounds')
    bounds = opts.bounds;
else
    bounds = [10, 0.001, 50;   % lower bounds
              100, 0.01, 500]; % upper bounds
end
lb = bounds(1,:);
ub = bounds(2,:);

nDim = numel(lb);

% Initialize swarm
rng('default');
X = rand(nParticles, nDim);           % normalized positions
V = zeros(nParticles, nDim);

pbest = X;
pbest_fitness = inf(nParticles, 1);

gbest = zeros(1, nDim);
gbest_fitness = inf;

% ACPSO parameters
w_max = 0.9;
w_min = 0.4;
c1 = 2;
c2 = 2;

curve = zeros(1, maxIter);

% Start parallel pool if needed
if isempty(gcp('nocreate'))
    parpool('local');
end

for iter = 1:maxIter
    w = w_max - (w_max - w_min) * iter / maxIter;

    gbest_candidates = zeros(nParticles, nDim);
    gbest_fit_candidates = inf(nParticles, 1);

    parfor i = 1:nParticles
        params_real = lb + X(i,:) .* (ub - lb);
        [fitness, ~, ~] = runLSTM(params_real, filename);
        if fitness < pbest_fitness(i)
            pbest(i,:) = X(i,:);
            pbest_fitness(i) = fitness;
        end

        gbest_candidates(i,:) = X(i,:);
        gbest_fit_candidates(i) = fitness;
    end

    [best_fit, idx] = min(gbest_fit_candidates);
    if best_fit < gbest_fitness
        gbest = gbest_candidates(idx,:);
        gbest_fitness = best_fit;
    end

    % Update velocity and position
    for i = 1:nParticles
        V(i,:) = w * V(i,:) ...
            + c1 * rand * (pbest(i,:) - X(i,:)) ...
            + c2 * rand * (gbest - X(i,:));
        X(i,:) = X(i,:) + V(i,:);
        X(i,:) = min(max(X(i,:), 0), 1);  % clip to [0, 1]
    end

    curve(iter) = gbest_fitness;
    fprintf('Iter %d/%d  Best RMSE: %.4f\n', iter, maxIter, gbest_fitness);
end

best_params = lb + gbest .* (ub - lb);
best_rmse = gbest_fitness;

fprintf('\n===== ACPSO Best Hyperparameters =====\n');
fprintf('Hidden units: %d\n', round(best_params(1)));
fprintf('Learning rate: %.5f\n', best_params(2));
fprintf('Epochs       : %d\n', round(best_params(3)));
fprintf('Minimum RMSE : %.4f\n', best_rmse);

figure;
plot(curve, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best RMSE');
title('ACPSO Convergence');
grid on;

% Shutdown the pool when done
delete(gcp('nocreate'));
end

function val = getOption(opts, name, default)
if isfield(opts, name)
    val = opts.(name);
else
    val = default;
end
end
