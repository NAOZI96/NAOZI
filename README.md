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

An example implementation of `runLSTM` is provided in
`matlab/runLSTM.m`.  The optimizer expects this function to accept a
parameter vector and a filename and to return the RMSE on the test set
used as the fitness value.
