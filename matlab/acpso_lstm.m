%% ACPSO_LSTM Main script
% This script optimizes LSTM hyperparameters using an adaptive PSO
% algorithm. Edit the configuration below and then run the script
% directly from MATLAB. It expects a companion function `runLSTM` that
% accepts a parameter vector and a filename and returns the RMSE used as
% the fitness value.

% -------- Configuration --------
filename   = '数据集.xlsx';      % data file
nParticles = 20;                % number of particles
maxIter    = 20;                % maximum iterations

% parameter bounds [hidden units, learning rate, epochs]
lb = [10, 0.001, 50];           % lower bounds
ub = [100, 0.01, 500];          % upper bounds

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

    fitness_all = zeros(nParticles, 1);
    gbest_candidates = zeros(nParticles, nDim);
    gbest_fit_candidates = inf(nParticles, 1);

    parfor i = 1:nParticles
        params_real = lb + X(i,:) .* (ub - lb);
        [fitness, ~, ~] = runLSTM(params_real, filename);

        fitness_all(i) = fitness;

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
