# The science behind neuro-sbi

This document explains, for a neuroscience audience, what this repository
infers, how, and — most importantly — how we check that the answer can be
trusted. Numbers from the committed run are in [RESULTS.md](RESULTS.md).

## 1. The inverse problem

A conductance-based model maps biophysical parameters $\theta$ (maximal
conductances, reversal potentials, kinetic offsets) to a voltage response $x$.
The forward direction is easy: integrate the ODEs. The scientifically
interesting direction is the inverse: *which parameters are consistent with a
recorded response, and how certain can we be?*

Classical approaches — hand-tuning, grid search, genetic algorithms — return
one or a few "best-fit" parameter sets. They do not say whether other,
very different parameter sets fit equally well, even though such degeneracy
is the rule rather than the exception in neural systems (Prinz, Bucher &
Marder 2004; Marder & Goaillard 2006). Bayesian inference answers the question
directly with the posterior

$$p(\theta \mid x_o) \propto p(x_o \mid \theta)\, p(\theta),$$

the full set of parameters compatible with the observation $x_o$ and the
prior knowledge $p(\theta)$. The obstacle is the likelihood
$p(x \mid \theta)$: for a stochastic simulator summarised by spike-based
features it has no closed form and cannot be evaluated.

**Simulation-based inference (SBI)** sidesteps the likelihood: it only
requires the ability to *simulate* (Cranmer, Brehmer & Louppe 2020).

## 2. The mechanistic model

A single-compartment Hodgkin–Huxley neuron (Hodgkin & Huxley 1952) in the
formulation of Pospischil et al. (2008) for regular-spiking cortical cells,
with Traub & Miles (1991) sodium and delayed-rectifier kinetics — the same
model Gonçalves et al. (2020) used to introduce neural posterior estimation to
neuroscience:

$$C_m \frac{dV}{dt} = -\bar g_\mathrm{Na}\, m^3 h\,(V-E_\mathrm{Na})
 - \bar g_\mathrm{K}\, n^4 (V-E_\mathrm{K})
 - \bar g_\mathrm{M}\, p\,(V-E_\mathrm{K})
 - g_\mathrm{leak}(V-E_\mathrm{leak}) + I_\mathrm{inj}(t) + C_m\,\sigma\,\xi(t).$$

* $m, h, n$ follow first-order kinetics whose rate functions are shifted by
  the threshold parameter $V_T$ (moving $V_T$ moves the spike threshold).
* $p$ is a slow, non-inactivating M-type K⁺ current ($\tau_p$ up to
  $\tau_\mathrm{max} = 600$ ms) that produces spike-frequency adaptation.
* $\xi(t)$ is Gaussian white noise with $\sigma = 0.1$ mV/√ms, representing
  channel and synaptic noise. It makes the simulator stochastic, so the
  likelihood is a proper (non-degenerate) distribution and repeated
  recordings of the same cell differ — as real recordings do.
* Stimulus protocol: 50 ms discarded burn-in after starting from the
  zero-current steady state, 20 ms baseline, a 100 ms step of 3 µA/cm²
  (≈ 0.46 nA into a 70 × 70 µm soma), 30 ms recovery. $C_m = 1$ µF/cm²,
  $E_\mathrm{Na} = 53$ mV, $E_\mathrm{K} = -107$ mV.

### Parameters and prior

| parameter | meaning | prior | reference value |
|---|---|---|---|
| $\bar g_\mathrm{Na}$ | transient Na⁺ conductance | log-uniform [5, 100] mS/cm² | 50 |
| $\bar g_\mathrm{K}$ | delayed-rectifier K⁺ conductance | log-uniform [0.5, 20] mS/cm² | 5 |
| $g_\mathrm{leak}$ | leak conductance | log-uniform [0.02, 0.5] mS/cm² | 0.1 |
| $\bar g_\mathrm{M}$ | slow M-type K⁺ conductance (adaptation) | log-uniform [0.005, 0.5] mS/cm² | 0.07 |
| $E_\mathrm{leak}$ | leak reversal potential | uniform [−85, −55] mV | −70 |
| $V_T$ | kinetic threshold offset | uniform [−70, −50] mV | −60 |

Conductances are strictly positive and span orders of magnitude, so their
prior is **log-uniform** (uniform in $\log_{10} g$): a priori a twofold change
is equally plausible at 1 and at 50 mS/cm². Inference is carried out in this
log space; the pairplot axes are therefore $\log_{10}$ conductances. The
reference values (Gonçalves et al. 2020) serve as the ground truth of the
worked recovery. Which parameters are inferred is configurable
(`src/neuro_sbi/params.py`); the rest are held at their reference values.

### Numerical integration (and how it is verified)

Training requires ~10⁵ simulations, so the integrator must be fast *and*
correct. We use an **exponential-Euler scheme**: gates are updated with the
Rush–Larsen formula (exact for frozen voltage; Rush & Larsen 1978), the
voltage with the exact solution of the linear membrane equation given the
gates, and the noise with Euler–Maruyama. The scheme is unconditionally stable
for the stiff sodium dynamics and is vectorised across neurons with numpy
(dt = 0.025 ms; ~2 ms per neuron per CPU core).

Correctness is not assumed: the test-suite compares it with an adaptive
**LSODA solution from `scipy.integrate.solve_ivp`** (rtol = atol = 1e-8) of the
deterministic system, requiring identical spike counts, spike times within
0.1 ms and all features within 2 %. It also checks that the analytically
computed resting state is an exact fixed point of the scheme, and that the
parallel simulator is bit-identical regardless of the number of processes.

## 3. Summary statistics

NPE could condition on the raw trace through a learned embedding network, but
for a step-current protocol a standard set of electrophysiological features is
more data-efficient and directly interpretable:

| feature | definition |
|---|---|
| spike count | APs during the step (peaks above −10 mV with ≥ 20 mV prominence) |
| resting potential (mean, SD) | over the 20 ms pre-stimulus baseline |
| stimulus V (mean, SD) | over the current step |
| first-spike latency | first AP peak − stimulus onset |
| firing rate | (n − 1)/(t_last − t_first) |
| AP threshold | V where dV/dt first exceeds 10 mV/ms before the first AP peak |
| AP amplitude | first-AP peak − AP threshold |
| AP width | full width at half amplitude of the first AP |
| AHP depth | AP threshold − minimum V before the next AP (or within 25 ms) |
| adaptation index | mean of (ISI_{k+1} − ISI_k)/(ISI_{k+1} + ISI_k) |

**Undefined features are never dropped.** Discarding simulations in which, for
example, no spike occurred would silently truncate the prior and bias the
posterior. Instead, undefined values get documented sentinels: latency is
censored at the stimulus duration, AP threshold at the maximal voltage reached
(the cell did *not* fire below it), and the spike-shape, rate and adaptation
features are 0. The spike count tells the network which values are sentinels.
The extractor is unit-tested against synthetic traces with analytically known
features.

A caveat that applies to every feature-based method: the posterior we compute
is $p(\theta \mid s(x))$, conditioned on the features $s(x)$, not on the full
trace. Unless the features are sufficient statistics (they are not) this
posterior is wider than $p(\theta \mid x)$ — it is correct, but it uses only
the information the features carry.

## 4. Neural posterior estimation

**Neural posterior estimation (NPE)** trains a conditional density estimator
$q_\phi(\theta \mid x)$ on simulated pairs $(\theta_i, x_i)$ with
$\theta_i \sim p(\theta)$ and $x_i \sim p(x \mid \theta_i)$, by maximising

$$\mathcal{L}(\phi) = \frac{1}{N}\sum_{i=1}^{N} \log q_\phi(\theta_i \mid x_i).$$

In expectation this equals
$-\mathbb{E}_{p(x)}\big[\mathrm{KL}\big(p(\theta\mid x)\,\|\,q_\phi(\theta\mid x)\big)\big] + \text{const}$,
so a sufficiently flexible $q_\phi$ converges to the true posterior **for
every x simultaneously** (Papamakarios & Murray 2016; Lueckmann et al. 2017;
Greenberg, Nonnenmacher & Macke 2019). This is *amortisation*: the cost is
paid once, during training; afterwards, inference for any new recording is a
single network evaluation — milliseconds, no new simulations, no MCMC. This is
what makes the 1000-observation calibration study below affordable.

The density estimator is a **normalising flow** (Papamakarios et al. 2021): an
invertible neural network that transforms a standard Gaussian into the
posterior, with weights that depend on $x$. The `full` configuration uses a
neural spline flow (NSF; Durkan et al. 2019: 5 rational-quadratic spline
coupling layers interleaved with learned linear mixing, 64 hidden units); the
CI configuration a masked autoregressive flow (MAF; Papamakarios, Pavlakou &
Murray 2017). Flows can
represent the skewed, banana-shaped and multimodal posteriors typical of
degenerate neural models.

**Ensembles against over-confidence.** A single flow trained on a finite
simulation budget is typically *over-confident*: its credible regions are too
small (Hermans et al. 2022). The cause is not a bug but estimation error. A
flow is fitted by maximum likelihood, and it is never told how uncertain its
own fit is.

We therefore train several flows that differ only in their random seed,
which sets the initial weights, the train/validation split and the order of
mini-batches. We use their equal-weight mixture as the posterior:

$$q(\theta \mid x) = \frac{1}{K}\sum_{k=1}^{K} q_k(\theta \mid x).$$

This is a *deep ensemble* (Lakshminarayanan, Pritzel & Blundell 2017). Where
the flows agree, the mixture is as sharp as each of them. Where they
disagree, the mixture covers all of their answers and is wider than any one
of them. The ensemble therefore puts the estimator's own uncertainty back
into the posterior.

The `full` configuration uses K = 5, the CI configuration K = 3. Sampling
from the mixture is exact: each draw picks a flow uniformly at random and
samples from it. Its density, which the coverage checks below need, is the
average of the flows' densities. The flows train in parallel processes, one
CPU thread each, and the result does not depend on how many run at once.

Implementation details (all via the `sbi` toolkit; Tejero-Cantero et al. 2020):

* Features are standardised, with the heavy-tailed resting-potential SD
  log-transformed first. This preprocessing is a module *inside* the network,
  so the saved posterior accepts raw feature vectors.
* 10 % of simulations are held out for early stopping: a training phase ends
  after 20 epochs without validation improvement and restores the best
  weights. In the `full` configuration training then resumes from those
  weights at a lower learning rate (5·10⁻⁴ → 10⁻⁴ → 2·10⁻⁵). The validation
  log-density of a single learning rate is too noisy for the stopping rule to
  see the slow, steady gains that the lower rates pick up.
* The flows have unbounded support while the prior is a box. Samples outside
  the prior are rejected, which is exactly the truncation the true posterior
  has. For each observation, proposals continue until enough samples are
  accepted.
* Every stage (training simulations, network seed, observation noise, SBC
  simulations, predictive simulations, posterior sampling) draws from its own
  seeded random stream, so the run is reproducible and the validation
  simulations are independent of the training set.

## 5. Is the posterior right? Validation

A trained network always returns *something*. A neuroscientist should accept a
posterior only after checks that would have caught a wrong one. We run four,
each answering a different question.

### 5.1 Simulation-based calibration (SBC): is the uncertainty honest?

Draw $\theta^* \sim p(\theta)$, simulate $x \sim p(x \mid \theta^*)$, then draw
$L$ samples from $q(\theta \mid x)$. If $q$ is the true posterior, $\theta^*$
is statistically indistinguishable from those samples, so the **rank** of
$\theta^*_j$ among the $L$ draws is uniform on $\{0,\dots,L\}$ — for every
parameter $j$ (Cook, Gelman & Rubin 2006; Talts et al. 2018). We repeat this
for 1000 independent simulations (never seen in training) with $L = 1000$.

How to read the rank histograms (`results/figures/sbc_ranks.png`):

| shape | meaning |
|---|---|
| flat, inside the grey 99 % band | calibrated |
| ∪-shaped (excess at both ends) | over-confident: posterior too narrow |
| ∩-shaped (deficit at both ends) | under-confident: posterior too wide |
| sloped / mass at one end | biased |

Uniformity is tested per parameter with a Kolmogorov–Smirnov test on the
randomised probability-integral transform $u = (r + U)/(L+1)$, which is
exactly $U(0,1)$ under the null hypothesis, and with a χ² test on the
histogram; we apply a Bonferroni correction over the six parameters. The ECDF
difference plot (`sbc_ecdf.png`) shows the same information without binning,
with a simultaneous 95 % Dvoretzky–Kiefer–Wolfowitz band (Massart 1990; see
Säilynoja, Bürkner & Vehtari 2022 for refinements).

**SBC is necessary but not sufficient.** The prior itself passes SBC
perfectly, and SBC averages over the prior, so a posterior can be calibrated
on average yet wrong for a particular recording. Hence the next checks.

### 5.2 Expected coverage: marginal and joint

For a calibrated posterior, the $\gamma$ credible region contains the truth in
a fraction $\gamma$ of trials (Hermans et al. 2022). `coverage.png` shows the
empirical coverage of (left) central marginal intervals for each parameter and
(right) **joint highest-posterior-density regions**, computed from the flow's
own density: for each test case we find the fraction of posterior draws with
higher density than $\theta^*$. The joint check also tests the *dependencies*
between parameters, which marginal SBC cannot see. Curves above the diagonal
indicate conservative (too wide) posteriors, curves below over-confident ones;
Hermans et al. (2022) argue that for scientific use, conservative is the
acceptable failure mode. For an ensemble, the joint panel also shows every
flow on its own (grey lines): the gap between them and the ensemble is what
the mixture adds.

### 5.3 Parameter recovery: does the data constrain the parameters?

* **Worked example.** We simulate a new recording from the reference cell
  (with a noise realisation never used in training), compute its posterior,
  and report for each parameter the posterior median, the 95 % credible
  interval, and the posterior percentile of the true value
  (`posterior_pairplot.png`). We also report the smallest joint HPD region
  that contains the truth.
* **Amortised recovery over many cells.** On the 1000 SBC simulations we
  report, per parameter: **posterior contraction**
  $1 - \mathrm{Var}_\text{post}/\mathrm{Var}_\text{prior}$ (0: the data taught us
  nothing, 1: the parameter is pinned down), the correlation between posterior
  mean and truth, normalised RMSE, and the posterior z-score
  $(\mathbb{E}_\text{post}[\theta] - \theta^*)/\mathrm{SD}_\text{post}$, whose
  distribution should be mean 0, SD 1 (Schad, Betancourt & Vasishth 2021).
  `recovery.png` plots posterior medians and 90 % intervals against the truth,
  separately for cells that fired and cells that stayed silent: a silent
  response carries almost no information about the spike-generating
  parameters, and a correct posterior must then fall back towards the prior
  rather than invent a confident answer.

Contraction is what separates a useful posterior from a calibrated but
uninformative one.

### 5.4 Posterior predictive checks (PPC): does the posterior reproduce the data?

Simulate the model at 500 posterior draws and compare with the observation
(Gelman, Meng & Stern 1996). `ppc_traces.png` shows voltage traces from
posterior draws next to traces from prior draws: the former should look like
the recording, the latter show how diverse the model class is.
`ppc_features.png` shows the predictive distribution of every feature with the
observed value; we report its predictive quantile $q$ (extreme values near 0
or 1 flag features the posterior fails to reproduce).

With synthetic data, the PPC tests the inference machinery. With real data it
becomes the primary test of **model adequacy**: if no parameter setting of the
model reproduces the recording, the posterior is conditioned on a
misspecified model, and SBC — which only uses simulated data — cannot detect
that.

## 6. Reading the posterior

Posterior correlations are often the scientific result. In this model, for
example, resting potential depends jointly on $E_\mathrm{leak}$,
$g_\mathrm{leak}$ and the M-current conductance (which is partially open at
rest), and the input resistance on all leak-like conductances; a recording can
therefore pin down a *combination* of them more tightly than each one alone,
which shows up as curved ridges in the pairplot. Such ridges are the
statistical signature of **compensation**: different conductance combinations
producing the same electrophysiology (Prinz et al. 2004; Gonçalves et al.
2020). A point estimate would hide them; the posterior exposes them.

The worked example in `posterior_pairplot.png` (axes zoomed to the posterior;
each diagonal panel states how much of the prior range it shows) shows both
regimes. The spike-generating parameters gNa, gK and VT are pinned to a few
per cent of their prior range. In contrast, gleak, gM and Eleak spread along
one curved ridge: less leak conductance is compensated by more M-current and
a more depolarised leak reversal, which keeps the resting potential and the
subthreshold input conductance unchanged. The recording constrains that
combination, not the individual values.

## 7. Limitations

* **Synthetic data.** All results here use simulated observations whose
  generating model is known. Real recordings add model misspecification
  (multi-compartment morphology, additional channels, temperature, recording
  artefacts); the PPC is the first line of defence there.
* **Features, not traces.** The posterior conditions on the summary
  statistics only (Section 3).
* **Protocol-specific amortisation.** The trained network is valid only for
  the stimulus protocol and noise model it was trained with (these are stored
  with the saved posterior and checked on reuse).
* **Prior dependence.** Weakly constrained parameters (low contraction) are
  dominated by their prior; their posteriors should not be over-interpreted.
* **Estimator error.** NPE is only asymptotically exact. Finite simulation
  budgets and finite training give approximation error, which is why the
  calibration checks are run and reported rather than assumed. Ensembling
  (Section 4) corrects the over-confidence of single flows, but it can
  overshoot into conservative (too wide) regions. In the committed `full`
  run the marginal intervals end up calibrated and the joint regions
  conservative. The effect is largest in spiking cells, where each flow's
  posterior is narrow and small disagreements between flows matter most
  ([RESULTS.md §3](RESULTS.md#3-calibration-sbc-and-expected-coverage)).
  Balanced NPE (Delaunoy et al. 2023) is an alternative that builds
  conservativeness into training. An ensemble of K flows costs K times the
  training of one; the flows train in parallel, so on K CPU cores the wall
  time is about that of a single flow.

## References

* Cook SR, Gelman A, Rubin DB (2006). Validation of software for Bayesian
  models using posterior quantiles. *J Comput Graph Stat* 15(3):675–692.
* **Cranmer K, Brehmer J, Louppe G (2020). The frontier of simulation-based
  inference. *PNAS* 117(48):30055–30062. doi:10.1073/pnas.1912789117**
* Delaunoy A, Miller BK, Forré P, Weniger C, Louppe G (2023). Balancing
  simulation-based inference for conservative posteriors. arXiv:2304.10978.
* Durkan C, Bekasov A, Murray I, Papamakarios G (2019). Neural spline flows.
  *NeurIPS* 32.
* Gelman A, Meng X-L, Stern H (1996). Posterior predictive assessment of model
  fitness via realized discrepancies. *Statistica Sinica* 6:733–807.
* **Gonçalves PJ, Lueckmann J-M, Deistler M, Nonnenmacher M, Öcal K,
  Bassetto G, Chintaluri C, Podlaski WF, Haddad SA, Vogels TP, Greenberg DS,
  Macke JH (2020). Training deep neural density estimators to identify
  mechanistic models of neural dynamics. *eLife* 9:e56261.
  doi:10.7554/eLife.56261**
* Greenberg DS, Nonnenmacher M, Macke JH (2019). Automatic posterior
  transformation for likelihood-free inference. *ICML*, PMLR 97:2404–2414.
* Hermans J, Delaunoy A, Rozet F, Wehenkel A, Begy V, Louppe G (2022). A trust
  crisis in simulation-based inference? Your posterior approximations can be
  unfaithful. *Transactions on Machine Learning Research*.
* Hodgkin AL, Huxley AF (1952). A quantitative description of membrane current
  and its application to conduction and excitation in nerve. *J Physiol*
  117(4):500–544.
* Lakshminarayanan B, Pritzel A, Blundell C (2017). Simple and scalable
  predictive uncertainty estimation using deep ensembles. *NeurIPS* 30.
* Lueckmann J-M, Gonçalves PJ, Bassetto G, Öcal K, Nonnenmacher M, Macke JH
  (2017). Flexible statistical inference for mechanistic models of neural
  dynamics. *NeurIPS* 30.
* Marder E, Goaillard J-M (2006). Variability, compensation and homeostasis in
  neuron and network function. *Nat Rev Neurosci* 7(7):563–574.
* Massart P (1990). The tight constant in the Dvoretzky–Kiefer–Wolfowitz
  inequality. *Ann Probab* 18(3):1269–1283.
* **Papamakarios G, Murray I (2016). Fast ε-free inference of simulation
  models with Bayesian conditional density estimation. *NeurIPS* 29.** (NPE)
* Papamakarios G, Pavlakou T, Murray I (2017). Masked autoregressive flow for
  density estimation. *NeurIPS* 30.
* Papamakarios G, Nalisnick E, Rezende DJ, Mohamed S, Lakshminarayanan B
  (2021). Normalizing flows for probabilistic modeling and inference. *JMLR*
  22(57):1–64.
* Pospischil M, Toledo-Rodriguez M, Monier C, Piwkowska Z, Bal T, Frégnac Y,
  Markram H, Destexhe A (2008). Minimal Hodgkin–Huxley type models for
  different classes of cortical and thalamic neurons. *Biol Cybern*
  99(4–5):427–441.
* Prinz AA, Bucher D, Marder E (2004). Similar network activity from disparate
  circuit parameters. *Nat Neurosci* 7(12):1345–1352.
* Rush S, Larsen H (1978). A practical algorithm for solving dynamic membrane
  equations. *IEEE Trans Biomed Eng* 25(4):389–392.
* Säilynoja T, Bürkner P-C, Vehtari A (2022). Graphical test for discrete
  uniformity and its applications in goodness-of-fit evaluation and multiple
  sample comparison. *Stat Comput* 32:32.
* Schad DJ, Betancourt M, Vasishth S (2021). Toward a principled Bayesian
  workflow in cognitive science. *Psychol Methods* 26(1):103–126.
* Talts S, Betancourt M, Simpson D, Vehtari A, Gelman A (2018). Validating
  Bayesian inference algorithms with simulation-based calibration.
  arXiv:1804.06788.
* Tejero-Cantero A, Boelts J, Deistler M, Lueckmann J-M, Durkan C,
  Gonçalves PJ, Greenberg DS, Macke JH (2020). sbi: A toolkit for
  simulation-based inference. *J Open Source Softw* 5(52):2505.
  doi:10.21105/joss.02505
* Traub RD, Miles R (1991). *Neuronal Networks of the Hippocampus.* Cambridge
  University Press.
