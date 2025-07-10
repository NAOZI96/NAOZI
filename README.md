# NAOZI

This repository contains example code for optimizing LSTM hyperparameters
using Adaptive Constrained Particle Swarm Optimization (ACPSO).  The MATLAB
implementation is located in `matlab/acpso_lstm.m`.

## Usage

1. Prepare a data file (e.g. `data.xlsx`) containing a single-column
   time series compatible with `runLSTM`.
2. Edit the configuration at the top of `acpso_lstm.m` to point to your
   data file and adjust particle/iteration settings if needed.
3. From MATLAB, run the script:

```matlab
cd matlab
acpso_lstm
```

The script prints the best hyperparameters and shows the convergence
curve.

An example implementation of `runLSTM` is provided in
`matlab/runLSTM.m`.  The optimizer expects this function to accept a
parameter vector and a filename and to return the RMSE on the test set
used as the fitness value.
