# Bulk analysis of constant-velocity optical tweezer experiments

This project is aimed to take large amounts of data from optical tweezer unfolding experiments,
and to perform as much automated analysis as is feasible.
The code is built on [lumicks pylake](https://github.com/lumicks/pylake) and assumes data is obtained from lumicks optical tweezer instruments,
but other sources can be used as well. The code is split up in a Curve class with supporting functions
and a notebook, which uses Curve objects to perform analysis.

Current functionality includes:
- Identifying protein unfolding events
- Identifying common experimental errors **(needs testing)**
  - Multiple tethers
  - Bead loss
- Computing unfolded domain contour lengths
- Computing unfolding forces

The data featured in the notebook file can be found here: https://harbor.lumicks.com/single-script/5586fed3-7e3b-4aa5-baba-c49548e3d54a
(requires a free account).
