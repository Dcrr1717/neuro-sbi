# neuro-sbi run report: `full` configuration

- Training simulations: **100,000** (65.8 % fire at least one AP)
- Density estimator: **NSF**, 5 transforms x 64 hidden units (125,126 weights); 499 epochs, best validation log q(θ|x) = 7.414
- Learning-rate annealing on plateau: 0.0005 (until epoch 122), 0.0001 (until epoch 434), 2e-05 (until epoch 499)
- Inferred parameters: log10 gNa [mS/cm²], log10 gK [mS/cm²], log10 gleak [mS/cm²], log10 gM [mS/cm²], Eleak [mV], VT [mV]
- Wall-clock [s]: simulate_training 82.4, train 5265.3, worked_recovery 3.1, ppc 5.0, sbc 186.2 (workers: 4)
- Training set and/or posterior loaded from cache; times are those of the original runs
- Versions: python 3.13.18, numpy 2.5.4, torch 2.15.0+cu132, sbi 0.27.0, matplotlib 3.11.2

## Worked recovery (held-out observation, known ground truth)

| parameter | truth | posterior median | 95 % credible interval | truth pct. | inside 95 % CI |
|---|---|---|---|---|---|
| gNa [mS/cm²] | 50.00 mS/cm² | 49.96 | [47.82, 52.55] | 51.2 | yes |
| gK [mS/cm²] | 5.00 mS/cm² | 5.04 | [4.97, 5.12] | 12.4 | yes |
| gleak [mS/cm²] | 0.1 mS/cm² | 0.0849 | [0.0356, 0.136] | 67.2 | yes |
| gM [mS/cm²] | 0.07 mS/cm² | 0.0846 | [0.0303, 0.119] | 30.9 | yes |
| Eleak [mV] | -70.00 mV | -69.44 | [-70.37, -65.58] | 21.9 | yes |
| VT [mV] | -60.00 mV | -59.81 | [-60.30, -59.34] | 22.7 | yes |

The ground truth lies inside the **14.3 %** joint highest-posterior-density region (smaller = closer to the mode).

## Posterior predictive check (summary statistics)

| feature | observed | posterior predictive mean ± sd | quantile q | prior predictive mean ± sd |
|---|---|---|---|---|
| spike_count | 5.00 | 4.98 ± 0.125 | 0.51 | 3.52 ± 4.01 |
| resting_mean | -70.58 | -70.33 ± 0.534 | 0.31 | -70.20 ± 12.21 |
| resting_std | 0.0994 | 0.144 ± 0.0506 | 0.17 | 0.504 ± 2.43 |
| stim_mean | -56.28 | -56.49 ± 0.434 | 0.59 | -52.24 ± 14.15 |
| stim_std | 19.39 | 19.44 ± 0.293 | 0.38 | 11.56 ± 8.59 |
| latency | 10.40 | 9.98 ± 1.23 | 0.65 | 44.65 ± 43.02 |
| firing_rate | 46.62 | 47.39 ± 1.75 | 0.38 | 44.93 ± 58.49 |
| ap_threshold | -47.10 | -47.06 ± 0.271 | 0.44 | -49.77 ± 11.02 |
| ap_amplitude | 97.07 | 97.07 ± 0.262 | 0.50 | 51.99 ± 42.41 |
| ap_width | 0.684 | 0.684 ± 0.00157 | 0.62 | 0.527 ± 0.476 |
| ahp_depth | 26.91 | 26.80 ± 0.256 | 0.66 | 10.77 ± 16.19 |
| adaptation_index | 0.0765 | 0.0852 ± 0.0203 | 0.27 | 0.019 ± 0.0714 |

Spike count reproduced exactly by 98 % of posterior-predictive simulations (prior predictive: 7 %).

## Simulation-based calibration (1000 simulations, L = 1000 posterior draws each)

| parameter | KS p | χ² p | uniform (Bonferroni α=0.05) | 50 % cov. | 90 % cov. | 95 % cov. |
|---|---|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.019 | 0.008 | yes | 0.453 | 0.875 | 0.920 |
| log10 gK [mS/cm²] | 0.404 | 0.515 | yes | 0.491 | 0.889 | 0.933 |
| log10 gleak [mS/cm²] | 0.009 | 0.007 | yes | 0.445 | 0.869 | 0.929 |
| log10 gM [mS/cm²] | 0.052 | 0.079 | yes | 0.466 | 0.869 | 0.915 |
| Eleak [mV] | 0.000 | 0.002 | **no** | 0.463 | 0.870 | 0.920 |
| VT [mV] | 0.487 | 0.729 | yes | 0.481 | 0.889 | 0.937 |

Joint HPD coverage: 50 % → 0.463, 80 % → 0.737, 90 % → 0.857, 95 % → 0.898 (KS test of HPD levels vs U(0,1): p = 0.000).
- spiking cells (n = 659): 50 % → 0.457, 80 % → 0.736, 90 % → 0.860, 95 % → 0.894
- silent cells (n = 341): 50 % → 0.475, 80 % → 0.739, 90 % → 0.850, 95 % → 0.906

## Amortized recovery on the same held-out simulations

| parameter | contraction | r(mean, truth) | NRMSE | z mean | z sd | 95 % coverage |
|---|---|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.706 | 0.833 | 0.556 | +0.033 | 1.052 | 0.934 |
| log10 gK [mS/cm²] | 0.698 | 0.833 | 0.558 | -0.024 | 1.032 | 0.943 |
| log10 gleak [mS/cm²] | 0.871 | 0.920 | 0.409 | +0.081 | 1.064 | 0.935 |
| log10 gM [mS/cm²] | 0.510 | 0.679 | 0.736 | +0.010 | 1.161 | 0.923 |
| Eleak [mV] | 0.936 | 0.960 | 0.286 | -0.154 | 1.173 | 0.928 |
| VT [mV] | 0.793 | 0.891 | 0.455 | -0.018 | 1.025 | 0.948 |

Posterior contraction split by response type (659 cells fired ≥ 1 AP, 341 stayed silent):

| parameter | contraction (spiking) | contraction (silent) | 95 % cov. (spiking) | 95 % cov. (silent) |
|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.998 | 0.141 | 0.935 | 0.933 |
| log10 gK [mS/cm²] | 0.999 | 0.117 | 0.945 | 0.938 |
| log10 gleak [mS/cm²] | 0.857 | 0.898 | 0.933 | 0.938 |
| log10 gM [mS/cm²] | 0.728 | 0.088 | 0.907 | 0.953 |
| Eleak [mV] | 0.945 | 0.917 | 0.920 | 0.944 |
| VT [mV] | 0.998 | 0.397 | 0.948 | 0.947 |
