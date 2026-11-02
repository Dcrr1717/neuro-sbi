# Results of the `full` run

All numbers below come from the committed run (`neuro-sbi run --config full`).
Its outputs are in [`results/`](../results):

* [`report.md`](../results/report.md): auto-generated tables
* [`metrics.json`](../results/metrics.json): every number
* [`sbc.npz`](../results/sbc.npz): per-simulation SBC ranks and HPD levels, for re-analysis
* [`posterior.pt`](../results/posterior.pt): the trained posterior

The method and how to read each diagnostic are explained in [SCIENCE.md](SCIENCE.md).

| | |
|---|---|
| training data | 100,000 simulations from the prior; 65.8 % fire at least one AP |
| estimator | neural spline flow, 5 transforms × 64 hidden units (125,126 weights) |
| training | 499 epochs in three learning-rate phases (5·10⁻⁴ → 10⁻⁴ → 2·10⁻⁵); best held-out log q(θ \| x) = 7.41 |
| wall clock (4 CPU cores) | simulation 82 s, training 88 min, all validation stages ≈ 3 min |
| validation data | 1000 new prior simulations (SBC, coverage, recovery) plus one held-out recording (worked example), never used in training |

## 1. Worked recovery: the ground truth lies inside the posterior

The observation is the reference cell (gNa = 50, gK = 5, gleak = 0.1,
gM = 0.07 mS/cm², Eleak = −70 mV, VT = −60 mV) simulated with a noise seed that
was never used in training.

| parameter | truth | posterior median | 95 % credible interval | inside |
|---|---|---|---|---|
| gNa [mS/cm²] | 50 | 49.96 | [47.82, 52.55] | yes |
| gK [mS/cm²] | 5 | 5.04 | [4.97, 5.12] | yes |
| gleak [mS/cm²] | 0.1 | 0.085 | [0.036, 0.136] | yes |
| gM [mS/cm²] | 0.07 | 0.085 | [0.030, 0.119] | yes |
| Eleak [mV] | −70 | −69.44 | [−70.37, −65.58] | yes |
| VT [mV] | −60 | −59.81 | [−60.30, −59.34] | yes |

All six true values lie inside their 95 % intervals. The full parameter vector
lies inside the 14.3 % highest-posterior-density region, so it sits close to the
posterior mode.

![Posterior pairplot](../results/figures/posterior_pairplot.png)

The spike-generating parameters are pinned down. The 95 % interval is about
±5 % for gNa, ±1.5 % for gK and ±0.5 mV for VT.

gleak, gM and Eleak are each only loosely determined. They spread along one
curved ridge: less leak is compensated by more M-current and a more
depolarised leak reversal. The recording fixes this combination rather than
the individual values, which is the compensation discussed in
[SCIENCE.md §6](SCIENCE.md#6-reading-the-posterior).

## 2. Posterior predictive check: the posterior reproduces the recording

![Posterior predictive voltage traces](../results/figures/ppc_traces.png)

Results from simulating the model at 500 posterior draws:

* **Spike count:** 98 % of the simulations fire exactly the observed 5 APs. With prior draws, 7 % do.
* **All 12 features:** the observed value falls at a central predictive quantile, between 0.17 (resting SD) and 0.66 (AHP depth). No feature is out of reach of the posterior.

See [`ppc_features.png`](../results/figures/ppc_features.png).

## 3. Calibration: SBC and expected coverage

Calibration is tested on 1000 independent prior simulations with 1000
posterior draws each.

![SBC rank histograms](../results/figures/sbc_ranks.png)

| parameter | SBC KS p | uniform (Bonferroni) | 50 % interval | 90 % interval | 95 % interval |
|---|---|---|---|---|---|
| log10 gNa | 0.019 | yes | 0.453 | 0.875 | 0.920 |
| log10 gK | 0.404 | yes | 0.491 | 0.889 | 0.933 |
| log10 gleak | 0.009 | yes | 0.445 | 0.869 | 0.929 |
| log10 gM | 0.052 | yes | 0.466 | 0.869 | 0.915 |
| Eleak | < 0.001 | **no** | 0.463 | 0.870 | 0.920 |
| VT | 0.487 | yes | 0.481 | 0.889 | 0.937 |

The interval columns give the empirical coverage of central credible intervals.
A calibrated posterior gives 0.50, 0.90 and 0.95.

Joint highest-posterior-density regions with nominal levels 50 / 80 / 90 / 95 %
contain the truth in 46.3 / 73.7 / 85.7 / 89.8 % of cases
([`coverage.png`](../results/figures/coverage.png)).

**Verdict: close to calibrated, but slightly over-confident.** The posterior
is a little too narrow. Marginal 95 % intervals cover the truth 91.5–93.7 % of
the time, and the joint 95 % region covers it 89.8 % of the time.

* **Eleak** fails the rank-uniformity test. It has an excess of high ranks
  ([`sbc_ecdf.png`](../results/figures/sbc_ecdf.png)): the posterior sits
  slightly below the truth (mean z-score −0.15) and is slightly too narrow
  (z-score SD 1.17).
* **gNa** and **gleak** show small excesses of low ranks. Both pass at the
  Bonferroni level, but only narrowly.
* **Where the deviations come from:**
  * Marginal deviations concentrate in the 659 cells that fired, where the
    posteriors are narrowest. Restricted to the 341 silent cells, all six
    per-parameter KS tests give p > 0.05 (computed from `sbc.npz`).
  * Joint over-confidence is similar in both groups: 95 % HPD coverage is
    0.894 for spiking cells and 0.906 for silent cells.
* **Probable cause:** mild overfitting. Training and held-out log-density
  separate after the first learning-rate decay, ending at 7.70 vs 7.36
  ([`training_curve.png`](../results/figures/training_curve.png)). Over-confidence
  from finite training is the usual failure mode of NPE (Hermans et al. 2022).
* **Practical consequence:** on this benchmark, 95 % empirical coverage needs
  central intervals with a nominal level of 95–97 % (depending on the
  parameter), or the nominal 97.5 % joint region. Ensembles of flows or balanced NPE would be the principled fix
  ([SCIENCE.md §7](SCIENCE.md#7-limitations)).

## 4. Amortized recovery: what the data constrain

![Parameter recovery on held-out simulations](../results/figures/recovery.png)

Posterior contraction is 1 − posterior variance / prior variance: 0 means the
data taught nothing, 1 means the parameter is pinned down. The figure plots
300 of the 1000 held-out cells.

| parameter | contraction (all) | spiking cells (659) | silent cells (341) | r(posterior mean, truth) |
|---|---|---|---|---|
| log10 gNa | 0.706 | 0.998 | 0.141 | 0.833 |
| log10 gK | 0.698 | 0.999 | 0.117 | 0.833 |
| log10 gleak | 0.871 | 0.857 | 0.898 | 0.920 |
| log10 gM | 0.510 | 0.728 | 0.088 | 0.679 |
| Eleak | 0.936 | 0.945 | 0.917 | 0.960 |
| VT | 0.793 | 0.998 | 0.397 | 0.891 |

In cells that fire, gNa, gK and VT are essentially identified (contraction
≥ 0.998). The overall averages for these parameters are modest only because a
third of prior draws produce no AP.

A silent response still constrains gleak and Eleak through the resting and
subthreshold voltage. It says little about the spike-generating parameters, and
for those the posterior correctly falls back towards the prior instead of
inventing a confident answer: 95 % coverage in silent cells is 0.93–0.95 for
every parameter.

gM is the hardest parameter. It acts slowly and trades off against leak (see
§1).

## 5. Effect of learning-rate annealing (first run vs committed run)

The first full run used a single learning rate. Its early stopping fired at
epoch 122, while the training curve was still rising. The committed run
resumes training from those weights at lower learning rates.

| | single learning rate (122 epochs) | annealed (499 epochs, committed) |
|---|---|---|
| best held-out log q(θ \| x) | 5.79 | **7.41** |
| joint HPD coverage at 50 / 95 % | 0.587 / 0.949 (conservative) | 0.463 / 0.898 (over-confident) |
| KS test of joint HPD levels | p ≈ 10⁻⁷ | p ≈ 2·10⁻⁴ |
| parameters failing SBC (Bonferroni) | gleak, gM | Eleak |
| contraction gleak / gM | 0.834 / 0.462 | 0.871 / 0.510 |

Annealing improved the held-out log-density by 1.6 nats per observation, which
is a large gain in expected KL divergence to the true posterior. It also moved
the joint calibration from conservative to slightly over-confident; by the KS
statistic it is closer to uniform than before. Neither estimator is perfectly
calibrated at n = 1000; the remaining errors are a few percentage points of
coverage. This trade-off is why the calibration checks are run on every
estimator rather than assumed.
