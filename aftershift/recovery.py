"""Budget-matched recovery: SAC fine-tuning under a shifted context vs. retrained oracle."""
from __future__ import annotations

import argparse
import csv
import json
import os

from stable_baselines3 import SAC

from .envs import CARTPOLE_NOMINAL, ContextCartPole
from .evaluate import evaluate_policy_at, load_model
from .metrics import i95



def parse_shift(s: str) -> dict:
    """Accept JSON ('{"pole_len": 2.0}') or shell-safe key=value syntax
    ('pole_len=2.0') -- PowerShell strips inner double quotes, which broke
    JSON args for one user run (see the project research log)."""
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        out = {}
        for kv in s.split(","):
            k, v = kv.split("=")
            out[k.strip()] = float(v)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run dir of the nominal SAC to fine-tune")
    ap.add_argument("--budgets", default="0,2000,10000,40000")
    ap.add_argument("--shift", default='{"pole_len": 2.0}', help="JSON context override")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--oracle", required=True, help="run dir of the retrained oracle")
    ap.add_argument("--out", default="results/recovery.csv")
    ap.add_argument("--summary", default="results/recovery_summary.json")
    args = ap.parse_args()

    shift = parse_shift(args.shift)
    ctx = dict(CARTPOLE_NOMINAL)
    ctx.update(shift)
    budgets = [int(b) for b in args.budgets.split(",")]
    name = os.path.basename(args.model.rstrip("/"))

    # oracle target
    oracle_model, _ = load_model(args.oracle)
    oracle_env = ContextCartPole(context=shift)
    ostats = evaluate_policy_at(oracle_model, oracle_env, ctx, args.episodes, base_seed=900_000)
    oracle_norm = ostats["mean_return"] / 500.0
    print(f"[recovery] oracle at {shift}: {ostats['mean_return']:.1f} (norm {oracle_norm:.3f})")

    rows = []
    for b in budgets:
        model, _ = load_model(args.model)  # fresh copy per budget (independent continuation)
        env = ContextCartPole(context=shift)
        model.set_env(env)
        buf = os.path.join(args.model, "replay_buffer.pkl")
        if os.path.exists(buf):
            model.load_replay_buffer(buf)
        if b > 0:
            model.learn(total_timesteps=b, reset_num_timesteps=False)
        st = evaluate_policy_at(model, env, ctx, args.episodes, base_seed=900_000)
        rows.append(dict(model=name, budget=b, mean_return=st["mean_return"],
                         norm_return=st["mean_return"] / 500.0))
        print(f"[recovery] {name} budget={b:>6}: {st['mean_return']:.1f} (norm {st['mean_return']/500.0:.3f})")

    oracle_label = f"oracle({os.path.basename(args.oracle.rstrip('/'))})"
    have_oracle = False
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                if row["model"] == oracle_label:
                    have_oracle = True
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "budget", "mean_return", "norm_return"])
        for r in rows:
            w.writerow([r["model"], r["budget"], r["mean_return"], r["norm_return"]])
        if not have_oracle:  # one oracle row per file (not per seed run)
            w.writerow([oracle_label, -1, ostats["mean_return"], oracle_norm])

    i95_v = i95([r["budget"] for r in rows], [r["norm_return"] for r in rows], oracle_norm)
    summary = dict(model=name, shift=shift, budgets=budgets,
                   returns=[r["norm_return"] for r in rows], oracle_norm_return=oracle_norm,
                   i95=i95_v)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[recovery] {name}: I95 = {i95_v} (oracle norm {oracle_norm:.3f})")


if __name__ == "__main__":
    main()
