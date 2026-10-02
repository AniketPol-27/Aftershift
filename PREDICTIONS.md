# AFTERSHIFT — Pilot Pre-Registration (v0.1)

**Written:** 2026-09-19, **before any training or evaluation runs were executed.**
**Status:** Frozen for the pilot. A fuller pre-registration (v1.0) will be written when the
protocol is frozen for scale-up (5 seeds, 20–50 eval episodes, more environments).

## Setting

Continuous-force CartPole (context-parameterized: `masscart`, `masspole`, `pole_len`,
`force_mag`, `cart_friction`, `gravity`; nominal values in `aftershift/envs.py`).
Fixed policies trained at nominal context (SAC, PPO), a domain-randomized SAC (DR ranges
in `aftershift/train.py`), a retrained oracle (SAC trained at `force_mag=4.0`), a
nominal-physics one-step residual monitor with a CUSUM detector, and budget-matched SAC
fine-tuning as the recovery baseline.

## Hypotheses (operationalized for the pilot)

- **H1 (robustness cliff).** For fixed policies trained at nominal context, normalized
  return (return / 500) stays near-max and then collapses abruptly: there exists at least
  one adjacent pair of sweep points with a mean-return drop ≥ 0.30 (on a 0–1 scale) for at
  least 2 of the 4 shift families (`masscart`↑, `force_mag`↓, `pole_len`↑, `cart_friction`↑).
  *Refuted if* degradation is gradual (max adjacent drop < 0.30) for ≥ 3 of 4 families.
- **H2 (DR shifts the cliff outward but steepens the collapse).** The DR-trained policy
  outperforms nominal-trained SAC at in-DR-range shifted points, but its largest adjacent
  drop beyond the DR range is ≥ the nominal-trained policy's largest adjacent drop.
  *Refuted if* DR degrades gradually where nominal-trained collapses, or DR never helps in-range.
- **H3 (silent failure of passive adaptation).** *Deferred to protocol scale* (requires the
  RMA-style history-encoder baseline, not in the pilot). Registered here to keep the
  hypothesis line honest.
- **H4 (detection lead time ≈ 0 for generic monitors).** A CUSUM on nominal-physics
  one-step prediction residuals, calibrated to ≤ 5% false alarms on nominal episodes,
  detects mid-episode shifts (applied at t = 100) with median lead time over control
  failure ≤ 25 steps, and fails to detect before termination in ≥ 30% of episodes that end
  in failure. Secondary prediction: detection is easier for actuator shifts (`force_mag`)
  than for inertial shifts (`masscart`).
  *Refuted if* median lead > 25 steps and pre-failure detection is near-universal.
- **H5 (recovery is not cheap for fine-tuning).** Budget-matched SAC fine-tuning under
  `force_mag` ×0.4 needs ≥ 10,000 post-shift interactions to reach 95% of the retrained
  oracle's normalized return. (This motivates the detect–identify–adapt MPC arm, which will
  claim order-of-magnitude fewer interactions; the pilot only measures the fine-tuning side.)

## Analysis plan (pilot)

- Deterministic policy evaluation; common random seeds across models (same initial states
  per sweep point); 5 episodes/point at pilot scale (raised at protocol scale).
- Report mean ± std across seeds per algorithm family; per-point failure rate and mean
  episode length in addition to return.
- Cliff metric: first sweep value (ordered away from nominal) with mean normalized return
  < 0.5; max adjacent drop.
- Detection: false-alarm calibration on 50 nominal episodes; 25 episodes per shift type;
  report detection rate, median/IQR lead time over failure, censored episodes marked.
- Recovery: budgets {0, 2k, 10k, 40k} × 2 seeds; I95 = smallest budget reaching 95% of
  oracle normalized return.
- All pilot numbers are **exploratory scale** and will not be quoted as final results
  without the protocol-scale rerun.

## Deviations log

Deviations from this registration, if any, will be recorded in `WORK_LOG.md` with reasons,
not silently applied.
