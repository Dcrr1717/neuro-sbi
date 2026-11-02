"""Auto-generated markdown report from ``metrics.json``."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt(v: float, digits: int = 3) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 1e-3:
        return f"{v:.{digits}g}"
    return f"{v:.{digits}g}" if abs(v) < 1 else f"{v:.{max(1, digits - 1)}f}"


def render_report(m: dict[str, Any]) -> str:
    cfg = m["config"]
    tr = m["training"]
    wr = m["worked_recovery"]
    lines: list[str] = []
    add = lines.append
    add(f"# neuro-sbi run report: `{cfg['name']}` configuration\n")
    add(
        f"- Training simulations: **{cfg['num_simulations']:,}** "
        f"({100 * m['training_data']['fraction_spiking']:.1f} % fire at least one AP)"
    )
    te = cfg["training"]
    add(
        f"- Density estimator: **{te['density_estimator'].upper()}**, "
        f"{te['num_transforms']} transforms x {te['hidden_features']} hidden units "
        f"({tr['num_parameters']:,} weights); {tr['epochs_trained']} epochs, "
        f"best validation log q(θ|x) = {-tr['best_validation_loss']:.3f}"
    )
    if len(tr.get("learning_rates", [])) > 1:
        phases = ", ".join(
            f"{lr:g} (until epoch {end})"
            for lr, end in zip(tr["learning_rates"], tr["phase_end_epochs"], strict=True)
        )
        add(f"- Learning-rate annealing on plateau: {phases}")
    add(f"- Inferred parameters: {', '.join(m['parameter_labels'])}")
    t = m["timings_sec"]
    add(
        "- Wall-clock [s]: " + ", ".join(f"{k} {v:.1f}" for k, v in t.items()) + f" (workers: {m['workers']})"
    )
    if m["training_data"].get("cached"):
        add("- Training set and/or posterior loaded from cache; times are those of the original runs")
    add(f"- Versions: {', '.join(f'{k} {v}' for k, v in m['versions'].items())}\n")

    add("## Worked recovery (held-out observation, known ground truth)\n")
    add("| parameter | truth | posterior median | 95 % credible interval | truth pct. | inside 95 % CI |")
    add("|---|---|---|---|---|---|")
    for s in wr["marginals"]:
        unit = s["unit"]
        lo, hi = s["ci95_physical"]
        add(
            f"| {s['label'].replace('log10 ', '')} | {_fmt(s['truth_physical'])} {unit} "
            f"| {_fmt(s['median_physical'])} | [{_fmt(lo)}, {_fmt(hi)}] "
            f"| {s['percentile']:.1f} | {'yes' if s['inside_ci95'] else '**no**'} |"
        )
    add(
        f"\nThe ground truth lies inside the **{100 * wr['hpd_level_of_truth']:.1f} %** "
        "joint highest-posterior-density region (smaller = closer to the mode).\n"
    )

    add("## Posterior predictive check (summary statistics)\n")
    add("| feature | observed | posterior predictive mean ± sd | quantile q | prior predictive mean ± sd |")
    add("|---|---|---|---|---|")
    for name, p in m["ppc"]["posterior"].items():
        pr = m["ppc"]["prior"][name]
        add(
            f"| {name} | {_fmt(p['observed'])} | {_fmt(p['pred_mean'])} ± {_fmt(p['pred_sd'])} "
            f"| {p['quantile']:.2f} | {_fmt(pr['pred_mean'])} ± {_fmt(pr['pred_sd'])} |"
        )
    add(
        f"\nSpike count reproduced exactly by {100 * m['ppc']['posterior_spike_count_exact_match']:.0f} % "
        f"of posterior-predictive simulations (prior predictive: "
        f"{100 * m['ppc']['prior_spike_count_exact_match']:.0f} %).\n"
    )

    sbc = m["sbc"]
    add(
        f"## Simulation-based calibration ({sbc['num_sbc']} simulations, "
        f"L = {sbc['posterior_samples']} posterior draws each)\n"
    )
    add("| parameter | KS p | χ² p | uniform (Bonferroni α=0.05) | 50 % cov. | 90 % cov. | 95 % cov. |")
    add("|---|---|---|---|---|---|---|")
    for j, (name, s) in enumerate(sbc["per_parameter"].items()):
        cov = sbc["marginal_coverage"]
        add(
            f"| {m['parameter_labels'][j]} | {s['ks_pvalue']:.3f} | {s['chi2_pvalue']:.3f} "
            f"| {'yes' if s['uniform_at_bonferroni_0.05'] else '**no**'} "
            f"| {cov['0.50'][name]:.3f} | {cov['0.90'][name]:.3f} | {cov['0.95'][name]:.3f} |"
        )
    jc = sbc["joint_hpd_coverage"]
    add(
        f"\nJoint HPD coverage: 50 % → {jc['0.50']:.3f}, 80 % → {jc['0.80']:.3f}, "
        f"90 % → {jc['0.90']:.3f}, 95 % → {jc['0.95']:.3f} "
        f"(KS test of HPD levels vs U(0,1): p = {sbc['joint_hpd_ks_pvalue']:.3f})."
    )
    for group, c in sbc.get("joint_hpd_coverage_by_group", {}).items():
        add(
            f"- {group} cells (n = {c['num']}): 50 % → {c['0.50']:.3f}, 80 % → {c['0.80']:.3f}, "
            f"90 % → {c['0.90']:.3f}, 95 % → {c['0.95']:.3f}"
        )
    add("")

    add("## Amortized recovery on the same held-out simulations\n")
    add("| parameter | contraction | r(mean, truth) | NRMSE | z mean | z sd | 95 % coverage |")
    add("|---|---|---|---|---|---|---|")
    for j, (_name, r) in enumerate(m["recovery"].items()):
        add(
            f"| {m['parameter_labels'][j]} | {r['contraction']:.3f} | {r['r']:.3f} | {r['nrmse']:.3f} "
            f"| {r['z_mean']:+.3f} | {r['z_std']:.3f} | {r['coverage95']:.3f} |"
        )
    groups = m.get("recovery_by_group", {})
    if len(groups) == 2:
        sp, si = groups["spiking"], groups["silent"]
        add(
            f"\nPosterior contraction split by response type "
            f"({sp['num']} cells fired ≥ 1 AP, {si['num']} stayed silent):\n"
        )
        add(
            "| parameter | contraction (spiking) | contraction (silent) "
            "| 95 % cov. (spiking) | 95 % cov. (silent) |"
        )
        add("|---|---|---|---|---|")
        for j, name in enumerate(m["recovery"]):
            a_, b_ = sp["per_parameter"][name], si["per_parameter"][name]
            add(
                f"| {m['parameter_labels'][j]} | {a_['contraction']:.3f} | {b_['contraction']:.3f} "
                f"| {a_['coverage95']:.3f} | {b_['coverage95']:.3f} |"
            )
    add("")
    return "\n".join(lines)


def write_report(metrics: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(render_report(metrics))
