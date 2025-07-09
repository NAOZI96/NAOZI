# NAOZI

This repository contains example code for optimizing LSTM hyperparameters
using Adaptive Constrained Particle Swarm Optimization (ACPSO).  The MATLAB
implementation is located in `matlab/acpso_lstm.m`.

## Usage

1. Prepare a data file (e.g. `data.xlsx`) compatible with your LSTM
   training function `runLSTM`.
2. From MATLAB, call the function:

```matlab
[params, rmse] = acpso_lstm('data.xlsx');
```

The function will return the best hyperparameters and display the
convergence curve.

`runLSTM` must be available on the MATLAB path and accept a parameter
vector and filename as input. It should return the RMSE used as the
fitness value for the optimizer.
