# neuro-sbi run report: `full` configuration

- Training simulations: **100,000** (65.8 % fire at least one AP)
- Density estimator: **NSF**, 5 transforms x 64 hidden units (125,126 weights); 122 epochs, best validation log q(θ|x) = 5.792
- Inferred parameters: log10 gNa [mS/cm²], log10 gK [mS/cm²], log10 gleak [mS/cm²], log10 gM [mS/cm²], Eleak [mV], VT [mV]
- Wall-clock [s]: simulate_training 82.4, train 1276.9, worked_recovery 2.4, ppc 5.0, sbc 116.0 (workers: 4)
- Versions: python 3.13.18, numpy 2.5.4, torch 2.15.0+cu132, sbi 0.27.0, matplotlib 3.11.2

## Worked recovery (held-out observation, known ground truth)

| parameter | truth | posterior median | 95 % credible interval | truth percentile | inside 95 % CI |
|---|---|---|---|---|---|
| gNa [mS/cm²] | 50.00 mS/cm² | 49.26 | [45.03, 54.41] | 61.9 | yes |
| gK [mS/cm²] | 5.00 mS/cm² | 5.04 | [4.88, 5.20] | 31.6 | yes |
| gleak [mS/cm²] | 0.1 mS/cm² | 0.0801 | [0.0266, 0.143] | 68.4 | yes |
| gM [mS/cm²] | 0.07 mS/cm² | 0.0844 | [0.0242, 0.119] | 32.2 | yes |
| Eleak [mV] | -70.00 mV | -69.30 | [-70.37, -64.61] | 19.8 | yes |
| VT [mV] | -60.00 mV | -59.81 | [-60.35, -59.28] | 25.1 | yes |

The ground truth lies inside the **8.5 %** joint highest-posterior-density region (smaller = closer to the mode).

## Posterior predictive check (summary statistics)

| feature | observed | posterior predictive mean ± sd | quantile q | prior predictive mean ± sd |
|---|---|---|---|---|
| spike_count | 5.00 | 4.99 ± 0.117 | 0.51 | 3.52 ± 4.01 |
| resting_mean | -70.58 | -70.33 ± 0.436 | 0.28 | -70.20 ± 12.21 |
| resting_std | 0.0994 | 0.145 ± 0.0531 | 0.17 | 0.504 ± 2.43 |
| stim_mean | -56.28 | -56.69 ± 0.536 | 0.71 | -52.24 ± 14.15 |
| stim_std | 19.39 | 19.47 ± 0.287 | 0.29 | 11.56 ± 8.59 |
| latency | 10.40 | 9.98 ± 1.46 | 0.63 | 44.65 ± 43.02 |
| firing_rate | 46.62 | 48.93 ± 1.92 | 0.10 | 44.93 ± 58.49 |
| ap_threshold | -47.10 | -47.05 ± 0.262 | 0.42 | -49.77 ± 11.02 |
| ap_amplitude | 97.07 | 97.00 ± 0.303 | 0.61 | 51.99 ± 42.41 |
| ap_width | 0.684 | 0.683 ± 0.00231 | 0.74 | 0.527 ± 0.476 |
| ahp_depth | 26.91 | 26.77 ± 0.318 | 0.66 | 10.77 ± 16.19 |
| adaptation_index | 0.0765 | 0.0795 ± 0.0208 | 0.33 | 0.019 ± 0.0714 |

Spike count reproduced exactly by 99 % of posterior-predictive simulations (prior predictive: 7 %).

## Simulation-based calibration (1000 simulations, L = 1000 posterior draws each)

| parameter | KS p | χ² p | uniform (Bonferroni α=0.05) | 50 % cov. | 90 % cov. | 95 % cov. |
|---|---|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.020 | 0.393 | yes | 0.485 | 0.905 | 0.946 |
| log10 gK [mS/cm²] | 0.170 | 0.726 | yes | 0.535 | 0.918 | 0.948 |
| log10 gleak [mS/cm²] | 0.000 | 0.005 | **no** | 0.463 | 0.878 | 0.925 |
| log10 gM [mS/cm²] | 0.005 | 0.060 | **no** | 0.458 | 0.876 | 0.920 |
| Eleak [mV] | 0.029 | 0.008 | yes | 0.449 | 0.875 | 0.914 |
| VT [mV] | 0.335 | 0.672 | yes | 0.494 | 0.900 | 0.946 |

Joint HPD coverage: 50 % → 0.587, 80 % → 0.852, 90 % → 0.919, 95 % → 0.949 (KS test of HPD levels vs U(0,1): p = 0.000).

## Amortized recovery on the same held-out simulations

| parameter | contraction | r(mean, truth) | NRMSE | z mean | z sd | 95 % coverage |
|---|---|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.711 | 0.832 | 0.558 | -0.081 | 0.981 | 0.955 |
| log10 gK [mS/cm²] | 0.699 | 0.833 | 0.559 | +0.002 | 0.941 | 0.957 |
| log10 gleak [mS/cm²] | 0.834 | 0.900 | 0.452 | +0.141 | 1.054 | 0.939 |
| log10 gM [mS/cm²] | 0.462 | 0.646 | 0.765 | -0.019 | 1.087 | 0.934 |
| Eleak [mV] | 0.928 | 0.957 | 0.295 | -0.083 | 1.092 | 0.918 |
| VT [mV] | 0.788 | 0.890 | 0.458 | -0.035 | 0.982 | 0.955 |
