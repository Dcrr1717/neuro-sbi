# neuro-sbi run report: `full` configuration

- Training simulations: **100,000** (65.8 % fire at least one AP)
- Density estimator: **equal-weight ensemble of 5 NSFs** differing only in their random seed, each 5 transforms x 64 hidden units (125,126 weights)
- Epochs per flow: 499, 633, 538, 517, 574; best validation log q(θ|x) per flow: 7.414, 8.857, 9.094, 7.494, 7.753
- Learning-rate annealing on plateau: 0.0005 → 0.0001 → 2e-05
- Inferred parameters: log10 gNa [mS/cm²], log10 gK [mS/cm²], log10 gleak [mS/cm²], log10 gM [mS/cm²], Eleak [mV], VT [mV]
- Wall-clock [s]: simulate_training 82.4, train 5170.2, worked_recovery 3.7, ppc 3.8, sbc 309.7 (workers: 5)
- Training set and/or posterior loaded from cache; times are those of the original runs
- Versions: python 3.13.18, numpy 2.5.4, torch 2.15.0+cu132, sbi 0.27.0, matplotlib 3.11.2

## Worked recovery (held-out observation, known ground truth)

| parameter | truth | posterior median | 95 % credible interval | truth pct. | inside 95 % CI |
|---|---|---|---|---|---|
| gNa [mS/cm²] | 50.00 mS/cm² | 50.47 | [47.63, 52.86] | 34.8 | yes |
| gK [mS/cm²] | 5.00 mS/cm² | 5.05 | [4.97, 5.14] | 10.2 | yes |
| gleak [mS/cm²] | 0.1 mS/cm² | 0.096 | [0.0374, 0.134] | 62.0 | yes |
| gM [mS/cm²] | 0.07 mS/cm² | 0.0741 | [0.0313, 0.119] | 35.6 | yes |
| Eleak [mV] | -70.00 mV | -69.74 | [-70.35, -66.05] | 20.5 | yes |
| VT [mV] | -60.00 mV | -59.83 | [-60.28, -59.39] | 22.5 | yes |

The ground truth lies inside the **19.9 %** joint highest-posterior-density region (smaller = closer to the mode).

## Posterior predictive check (summary statistics)

| feature | observed | posterior predictive mean ± sd | quantile q | prior predictive mean ± sd |
|---|---|---|---|---|
| spike_count | 5.00 | 4.98 ± 0.14 | 0.51 | 3.52 ± 4.01 |
| resting_mean | -70.58 | -70.47 ± 0.352 | 0.39 | -70.20 ± 12.21 |
| resting_std | 0.0994 | 0.143 ± 0.05 | 0.19 | 0.504 ± 2.43 |
| stim_mean | -56.28 | -56.35 ± 0.38 | 0.50 | -52.24 ± 14.15 |
| stim_std | 19.39 | 19.38 ± 0.301 | 0.42 | 11.56 ± 8.59 |
| latency | 10.40 | 10.26 ± 1.01 | 0.51 | 44.65 ± 43.02 |
| firing_rate | 46.62 | 47.09 ± 1.72 | 0.47 | 44.93 ± 58.49 |
| ap_threshold | -47.10 | -47.07 ± 0.271 | 0.45 | -49.77 ± 11.02 |
| ap_amplitude | 97.07 | 97.07 ± 0.271 | 0.49 | 51.99 ± 42.41 |
| ap_width | 0.684 | 0.684 ± 0.00167 | 0.63 | 0.527 ± 0.476 |
| ahp_depth | 26.91 | 26.83 ± 0.261 | 0.61 | 10.77 ± 16.19 |
| adaptation_index | 0.0765 | 0.0828 ± 0.0183 | 0.27 | 0.019 ± 0.0714 |

Spike count reproduced exactly by 98 % of posterior-predictive simulations (prior predictive: 7 %).

## Simulation-based calibration (1000 simulations, L = 1000 posterior draws each)

| parameter | KS p | χ² p | uniform (Bonferroni α=0.05) | 50 % cov. | 90 % cov. | 95 % cov. |
|---|---|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.056 | 0.004 | yes | 0.524 | 0.933 | 0.962 |
| log10 gK [mS/cm²] | 0.108 | 0.007 | yes | 0.537 | 0.919 | 0.959 |
| log10 gleak [mS/cm²] | 0.065 | 0.072 | yes | 0.535 | 0.926 | 0.959 |
| log10 gM [mS/cm²] | 0.077 | 0.180 | yes | 0.525 | 0.924 | 0.961 |
| Eleak [mV] | 0.000 | 0.020 | **no** | 0.516 | 0.911 | 0.948 |
| VT [mV] | 0.915 | 0.850 | yes | 0.505 | 0.915 | 0.951 |

Joint HPD coverage: 50 % → 0.696, 80 % → 0.898, 90 % → 0.952, 95 % → 0.982 (KS test of HPD levels vs U(0,1): p = 0.000).
- spiking cells (n = 659): 50 % → 0.754, 80 % → 0.924, 90 % → 0.971, 95 % → 0.989
- silent cells (n = 341): 50 % → 0.584, 80 % → 0.848, 90 % → 0.915, 95 % → 0.968

### Ensemble vs. its individual flows (same simulations)

| posterior | held-out log q(θ*\|x) | joint HPD coverage 50 / 80 / 90 / 95 % | marginal 95 % coverage (min–max) | parameters passing SBC |
|---|---|---|---|---|
| flow 1 | 7.519 | 0.464 / 0.737 / 0.855 / 0.907 | 0.925–0.946 | 4 of 6 |
| flow 2 | 8.856 | 0.419 / 0.725 / 0.840 / 0.907 | 0.938–0.943 | 6 of 6 |
| flow 3 | 8.955 | 0.438 / 0.725 / 0.834 / 0.905 | 0.936–0.949 | 6 of 6 |
| flow 4 | 7.375 | 0.437 / 0.722 / 0.839 / 0.907 | 0.932–0.943 | 2 of 6 |
| flow 5 | 7.622 | 0.443 / 0.715 / 0.817 / 0.900 | 0.922–0.943 | 3 of 6 |
| **ensemble** | 8.797 | 0.696 / 0.898 / 0.952 / 0.982 | 0.948–0.962 | 5 of 6 |

Held-out log q(θ*|x): mean log-density of the true parameters under the (untruncated) flow or mixture; higher is better.

## Amortized recovery on the same held-out simulations

| parameter | contraction | r(mean, truth) | NRMSE | z mean | z sd | 95 % coverage |
|---|---|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.708 | 0.835 | 0.553 | +0.026 | 0.878 | 0.962 |
| log10 gK [mS/cm²] | 0.702 | 0.836 | 0.554 | -0.001 | 0.900 | 0.959 |
| log10 gleak [mS/cm²] | 0.881 | 0.950 | 0.329 | +0.039 | 0.854 | 0.959 |
| log10 gM [mS/cm²] | 0.529 | 0.766 | 0.649 | -0.020 | 0.861 | 0.961 |
| Eleak [mV] | 0.939 | 0.968 | 0.256 | -0.084 | 1.004 | 0.948 |
| VT [mV] | 0.795 | 0.892 | 0.453 | -0.013 | 0.954 | 0.950 |

Posterior contraction split by response type (659 cells fired ≥ 1 AP, 341 stayed silent):

| parameter | contraction (spiking) | contraction (silent) | 95 % cov. (spiking) | 95 % cov. (silent) |
|---|---|---|---|---|
| log10 gNa [mS/cm²] | 0.999 | 0.146 | 0.976 | 0.935 |
| log10 gK [mS/cm²] | 0.999 | 0.126 | 0.971 | 0.935 |
| log10 gleak [mS/cm²] | 0.873 | 0.895 | 0.967 | 0.944 |
| log10 gM [mS/cm²] | 0.741 | 0.119 | 0.959 | 0.965 |
| Eleak [mV] | 0.949 | 0.920 | 0.947 | 0.950 |
| VT [mV] | 0.998 | 0.403 | 0.958 | 0.935 |
