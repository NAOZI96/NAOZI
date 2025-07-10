function [best_params, best_rmse, curve] = acpso_lstm(filename, opts)
% ACPSO_LSTM Optimize LSTM hyperparameters using an adaptive PSO.
% ACPSO_LSTM 使用自适应粒子群算法优化 LSTM 超参数。
%   [BEST_PARAMS, BEST_RMSE, CURVE] = ACPSO_LSTM(FILENAME) runs the
%   adaptive particle swarm optimizer on data stored in FILENAME using the
%   external function runLSTM. It returns the best hyperparameters,
%   the associated RMSE, and the convergence curve.
%   [BEST_PARAMS, BEST_RMSE, CURVE] = ACPSO_LSTM(FILENAME) 会在数据文件
%   FILENAME 上运行自适应粒子群优化算法，并调用外部函数 runLSTM。函数
%   返回最佳超参数、对应的 RMSE 以及收敛曲线。
%
%   opts is a structure with optional fields:
%   opts 为包含以下可选字段的结构体：
%       .nParticles  - number of particles (default 20)
%       .nParticles  - 粒子数量（默认 20）
%       .nIter       - maximum iterations (default 20)
%       .nIter       - 最大迭代次数（默认 20）
%       .bounds      - 2xD matrix of [lb; ub] for each dimension
%       .bounds      - 每个维度 [下界; 上界] 的 2xD 矩阵
%
%   The optimizer searches for three parameters:
%   优化器搜索三个参数：
%       hidden units, learning rate, training epochs.
%       隐藏单元数、学习率、训练周期。
%
%   Example:
%   示例：
%       [p, rmse] = acpso_lstm('data.xlsx');
%
%   NOTE: This function expects a companion function RUNLSTM that accepts
%   the real-valued parameter vector and dataset filename, and returns the
%   RMSE on the validation or test set.
%   注意：本函数依赖同目录下的 RUNLSTM，它接收实值参数向量和数据文件名，
%   返回验证集或测试集上的 RMSE。

if nargin < 2
    opts = struct();
end

% Default options
% 默认选项
nParticles = getOption(opts, 'nParticles', 20);
maxIter     = getOption(opts, 'nIter', 20);

% Parameter bounds: [hidden units, learning rate, epochs]
% 参数范围：[隐藏单元数、学习率、训练周期]
if isfield(opts, 'bounds')
    bounds = opts.bounds;
else
    bounds = [10, 0.001, 50;   % lower bounds
              100, 0.01, 500]; % upper bounds
    % 下界与上界
end
lb = bounds(1,:);
ub = bounds(2,:);

nDim = numel(lb);

% Initialize swarm
% 初始化粒子群
rng('default');
X = rand(nParticles, nDim);           % normalized positions
% 归一化位置
V = zeros(nParticles, nDim);

pbest = X;
pbest_fitness = inf(nParticles, 1);

gbest = zeros(1, nDim);
gbest_fitness = inf;

% ACPSO parameters
% ACPSO 参数
w_max = 0.9;
w_min = 0.4;
c1 = 2;
c2 = 2;

curve = zeros(1, maxIter);

% Start parallel pool if needed
% 如有需要，启动并行池
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
    % 更新粒子速度和位置
    for i = 1:nParticles
        V(i,:) = w * V(i,:) ...
            + c1 * rand * (pbest(i,:) - X(i,:)) ...
            + c2 * rand * (gbest - X(i,:));
        X(i,:) = X(i,:) + V(i,:);
        X(i,:) = min(max(X(i,:), 0), 1);  % clip to [0, 1]
        % 限制在 [0, 1] 区间内
    end

    curve(iter) = gbest_fitness;
    fprintf('Iter %d/%d  Best RMSE: %.4f\n', iter, maxIter, gbest_fitness);
    % 记录并输出当前最佳 RMSE
end

best_params = lb + gbest .* (ub - lb);
best_rmse = gbest_fitness;

fprintf('\n===== ACPSO Best Hyperparameters =====\n');
% 输出最佳超参数
fprintf('Hidden units: %d\n', round(best_params(1)));
fprintf('Learning rate: %.5f\n', best_params(2));
fprintf('Epochs       : %d\n', round(best_params(3)));
fprintf('Minimum RMSE : %.4f\n', best_rmse);

figure;
% 绘制收敛曲线
plot(curve, 'r-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best RMSE');
title('ACPSO Convergence');
grid on;

% Shutdown the pool when done
% 任务完成后关闭并行池
delete(gcp('nocreate'));
end

function val = getOption(opts, name, default)
% GETOPTION Retrieve value from a struct with a default.
% GETOPTION 从结构体中读取字段值，如不存在则返回默认值。
if isfield(opts, name)
    val = opts.(name);
else
    val = default;
end
end
