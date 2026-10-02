# AFTERSHIFT

**A benchmark and evaluation protocol for post-deployment dynamics shifts in continuous control.**

*Aniket Pol and Jasnoor Kaur — Indian Institute of Technology Bombay*

Deployed controllers face wear, payloads, and weakened actuators: the dynamics change
*after* training, while the controller is frozen. RL generalization research evaluates
train-time context variation and zero-shot transfer, but not the deployment-time
**shift event** itself. AFTERSHIFT makes that event the object of study, with three
evaluation axes over the same mid-episode shift on a context-parameterized cart-pole
(geometry, inertia, actuation, dissipation):

- **Degradation** — robustness curves and cliff points as physical parameters move
  away from nominal, per controller family (SAC, PPO, domain-randomized SAC).
- **Detection** — a false-alarm-calibrated CUSUM monitor on nominal-physics residuals:
  detection rates, pre-failure alarms, and lead time over failure.
- **Recovery** — budget-matched restoration after the shift (frozen, fine-tuning,
  retraining, and a classical detect → identify → adapt stack), scored against an
  analytic LQR oracle with every post-shift interaction charged.

## Headline results

| Finding | Result |
|---|---|
| Robustness rankings **invert** across shift axes | PPO is the most robust family on mass, force, and friction (0.82–1.00 at the severe ends) but collapses on pole length (≤0.08 beyond ×3); DR-SAC wins geometry (0.654 at ×5) |
| Warnings exist; frozen policies cannot use them | 100% detection of geometry/mass shifts with 202–298 steps of median lead time; one shift shows a 298-step verified warning while 73% of episodes still fail |
| Detection has an identifiability blind spot | Friction shifts: 6–15% detection (the changed term is not excited under a stabilizing policy) |
| Measurement beats gradient search at recovery | Self-ID: **546 post-shift interactions to oracle-level performance, 10/10 seeds**; fine-tuning: 5,000 interactions on 1/3 seeds, non-monotonic; retraining fails the convergence gate at 275k |
| Identification error gates recovery | Spearman ρ = −0.74 between estimation error and post-adaptation return (n=50); a sharp sufficiency tolerance (≈±0.75 m at 2.0 m) |

All numbers above are reproducible from this repository: `run_all.py` re-runs every
experiment from a fresh clone, the archived raw data of record is in
`results/raw_user_run/`, and `scripts/make_latex_figs.py` regenerates every figure
and table from that archive (with data assertions at build time).

## Quickstart

CPU-only; Python 3.10+ (developed and timed on Python 3.12, Windows 11, laptop CPU).

```bash
pip install -r requirements.txt
python run_all.py --stage validate    # ~2-min correctness gate
python run_all.py --stage all         # full battery (~115 min on a laptop CPU)
```

Stages: `validate → train → sweep → detection → recovery → selfid → plots → package`
(resume-safe; models in `runs/`, results in `results/`). The validate gate must pass
before any experiment stage. To regenerate the report's figures and tables from the
archived data: `python scripts/make_latex_figs.py`.

## Repository layout

```
run_all.py                   one-command, cross-platform experiment driver
aftershift/                  package: envs, shifts, detectors, train, evaluate,
                             recovery, selfid, converge, metrics, plots
scripts/make_latex_figs.py   regenerates all paper figures/tables from raw data
results/raw_user_run/        archived raw data behind every published number
                             (CSV + metadata, verified on the deployment machine)
PREDICTIONS.md               timestamped pre-registration (written before any runs)
```

## Reproducibility

Every number in the report is generated from `results/raw_user_run/` by script, with
data assertions at build time. Pre-registered hypotheses are frozen in
`PREDICTIONS.md` (written before any training or evaluation runs) and scored in the
paper's Appendix A; retractions (one pilot finding failed replication and is publicly
retracted) and protocol deviations are documented in the paper's appendices. Writers
are idempotent, so interrupted runs can be resumed without duplicating rows.

## Citation

```bibtex
@misc{pol2026aftershift,
  title  = {AFTERSHIFT: A Benchmark and Evaluation Protocol for
            Post-Deployment Dynamics Shifts in Continuous Control},
  author = {Pol, Aniket and Kaur, Jasnoor},
  url    = {https://github.com/AniketPol-27/Aftershift},
  year   = {2026}
}
```
