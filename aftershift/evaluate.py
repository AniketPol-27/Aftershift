"""Evaluation: degradation sweeps and mid-episode-shift detection experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from stable_baselines3 import SAC, PPO

from .detectors import CUSUMDetector, CartPolePhysicsPredictor, calibrate_cusum, residual_norm
from .envs import CARTPOLE_NOMINAL, ContextCartPole
from .metrics import lead_time_stats
from .shifts import ShiftScheduler, step_events

# Pilot sweep grid (v0.2 — extended after v0.1 showed SAC survives the initial ranges;
# see research-log deviation D9). `mult` = multiplier on nominal; `abs` = absolute value.
# Ordered from nominal outward (callers rely on this for cliff metrics).
SWEEPS = {
    "masscart": ("mult", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]),
    "force_mag": ("mult", [1.0, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]),
    "pole_len": ("mult", [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    "cart_friction": ("abs", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]),
}

# Mid-episode detection shifts. First three are control-BENIGN for SAC (policies survive);
# pole_len 2.0 is control-DEGRADING (norm return ~0.42) — added in deviation D10 so that
# detection lead-time over failure is measurable, and to probe the detection/degradation
# decoupling on the benign shifts.
PILOT_SHIFTS = [("force_mag", 5.0), ("masscart", 3.0), ("cart_friction", 2.0), ("pole_len", 2.0)]


def load_model(run_dir: str):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    cls = SAC if meta["algo"] == "sac" else PPO
    model = cls.load(os.path.join(run_dir, "model.zip"))
    return model, meta


def evaluate_policy_at(model, env, ctx: dict, episodes: int, base_seed: int) -> dict:
    """Deterministic evaluation under a fixed context `ctx` (full context dict)."""
    rets, lens, fails = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        env.set_context(**ctx)
        done, R, L, fail = False, 0.0, 0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            R += float(r)
            L += 1
            fail = fail or term
            done = term or trunc
        rets.append(R)
        lens.append(L)
        fails.append(fail)
    return dict(mean_return=float(np.mean(rets)), std_return=float(np.std(rets)),
                mean_len=float(np.mean(lens)), fail_rate=float(np.mean(fails)),
                episodes=episodes)


# ---------------- degradation sweep ----------------

def cmd_degradation(args):
    header = ["model", "param", "value", "mean_return", "std_return", "fail_rate",
              "mean_len", "norm_return", "episodes"]
    # idempotent re-runs: skip (model, param, value) rows already in the file
    # (fixes the duplicate-row artifact from double-run batteries)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["param"], float(row["value"])))
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for run_dir in args.models:
            model, _ = load_model(run_dir)
            name = os.path.basename(run_dir.rstrip("/"))
            for param, (kind, vals) in SWEEPS.items():
                for v in vals:
                    ctx = dict(CARTPOLE_NOMINAL)
                    ctx[param] = round(v * ctx[param], 6) if kind == "mult" else v
                    if (name, param, ctx[param]) in done:
                        continue
                    env = ContextCartPole()
                    st = evaluate_policy_at(model, env, ctx, args.episodes, base_seed=555_000)
                    w.writerow([name, param, ctx[param], st["mean_return"], st["std_return"],
                                st["fail_rate"], st["mean_len"], st["mean_return"] / 500.0,
                                args.episodes])
            print(f"[degradation] {name} done")
    print(f"[degradation] wrote {args.out}")


# ---------------- detection ----------------

def _rollout_residuals(model, env, predictor, episodes, base_seed):
    """Residual streams on NOMINAL episodes (for CUSUM calibration)."""
    out = []
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        env.set_context(**CARTPOLE_NOMINAL)
        res, done = [], False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            pred = predictor.predict(obs, a)
            obs, r, term, trunc, _ = env.step(a)
            res.append(residual_norm(pred, obs))
            done = term or trunc
        out.append(res)
    return out


def _detection_episode(model, env, predictor, cusum, events, base_seed, ep_idx):
    """One episode with a mid-episode shift; monitor runs alongside the policy."""
    obs, _ = env.reset(seed=base_seed + ep_idx)
    env.set_context(**CARTPOLE_NOMINAL)
    sched = ShiftScheduler(events)
    sched.reset()
    cusum.reset()
    detected, t_det, t_fail = False, None, None
    for t in range(env.max_steps):
        sched.apply(env, t)
        a, _ = model.predict(obs, deterministic=True)
        pred = predictor.predict(obs, a)
        obs, r, term, trunc, _ = env.step(a)
        if not detected and cusum.update(residual_norm(pred, obs)):
            detected, t_det = True, t
        if term:
            t_fail = t
            break
        if trunc:
            break
    return dict(detected=detected, t_det=t_det, t_fail=t_fail, length=t + 1)


def cmd_detection(args):
    model, _ = load_model(args.model)
    env = ContextCartPole()
    predictor = CartPolePhysicsPredictor()

    cal = _rollout_residuals(model, env, predictor, episodes=args.cal_episodes, base_seed=700_000)
    nu, k, h, fpr = calibrate_cusum(cal, target_fpr=0.05)
    print(f"[detection] CUSUM calibrated: nu={nu:.4f} k={k:.4f} h={h:.3f} (in-sample FA={fpr:.3f})")

    summary_rows, episode_rows = [], []
    for param, val in PILOT_SHIFTS:
        events = step_events(args.t_shift, **{param: val})
        leads, n_det, n_fail, n_fail_det, n_surv = [], 0, 0, 0, 0
        for i in range(args.episodes):
            cusum = CUSUMDetector(nu, k, h)
            out = _detection_episode(model, env, predictor, cusum, events, 800_000, i)
            det, fail = out["detected"], out["t_fail"] is not None
            n_det += det
            n_fail += fail
            if fail:
                n_fail_det += det and out["t_det"] <= out["t_fail"]
                if det and out["t_det"] <= out["t_fail"]:
                    leads.append(out["t_fail"] - out["t_det"])
            else:
                n_surv += 1
            episode_rows.append([param, val, i, det, out["t_det"], out["t_fail"], out["length"]])
        ls = lead_time_stats(leads)
        summary_rows.append([
            os.path.basename(args.model.rstrip("/")), param, val, args.t_shift, args.episodes,
            n_det / args.episodes, n_fail / args.episodes,
            (n_fail_det / n_fail) if n_fail else None,
            ls["median"], ls["iqr"], ls["n"], n_surv,
        ])
        print(f"[detection] {param}->{val}: detect={n_det}/{args.episodes} fail={n_fail} "
              f"pre-fail detect={n_fail_det}/{n_fail if n_fail else 0} median_lead={ls['median']}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "param", "shift_to", "t_shift", "episodes", "detect_rate",
                    "fail_rate", "prefail_detect_rate", "median_lead", "iqr_lead",
                    "n_leads", "n_survived"])
        w.writerows(summary_rows)
    with open(args.out.replace(".csv", "_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "shift_to", "ep", "detected", "t_det", "t_fail", "length"])
        w.writerows(episode_rows)
    print(f"[detection] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("degradation")
    d.add_argument("--models", nargs="+", required=True)
    d.add_argument("--episodes", type=int, default=5)
    d.add_argument("--out", default="results/degradation.csv")

    t = sub.add_parser("detection")
    t.add_argument("--model", required=True)
    t.add_argument("--episodes", type=int, default=25)
    t.add_argument("--cal-episodes", type=int, default=50)
    t.add_argument("--t-shift", type=int, default=100)
    t.add_argument("--out", default="results/detection.csv")

    args = ap.parse_args()
    if args.cmd == "degradation":
        cmd_degradation(args)
    else:
        cmd_detection(args)


if __name__ == "__main__":
    main()
