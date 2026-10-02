"""Continue training a run until its nominal-context eval return reaches a threshold.

Baseline convergence rule (decision D7 in the project research log): every fixed-policy baseline must
reach >= `--threshold` return at its training context before entering degradation
analysis; extra training is applied in `--chunk` steps up to `--max-extra`.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from stable_baselines3 import SAC, PPO

from .envs import CARTPOLE_NOMINAL
from .evaluate import evaluate_policy_at
from .train import build_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, default=450.0)
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--max-extra", type=int, default=75000)
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    with open(os.path.join(args.run, "meta.json")) as f:
        meta = json.load(f)
    assert meta["algo"] in ("sac", "ppo"), "unsupported algo"

    ctx = dict(CARTPOLE_NOMINAL)
    ctx.update(meta["context"])
    # continue training in the SAME regime the run was trained in (DR env for DR runs)
    env = build_env(meta["env"], meta["context"], dr=bool(meta.get("dr", False)))
    eval_env = build_env(meta["env"], meta["context"], dr=False)

    cls = SAC if meta["algo"] == "sac" else PPO
    model = cls.load(os.path.join(args.run, "model.zip"))
    if meta["algo"] == "sac":
        buf = os.path.join(args.run, "replay_buffer.pkl")
        if os.path.exists(buf):
            model.load_replay_buffer(buf)
    model.set_env(env)

    extra = 0
    st = evaluate_policy_at(model, eval_env, ctx, args.episodes, base_seed=123_000)
    while st["mean_return"] < args.threshold and extra < args.max_extra:
        model.learn(total_timesteps=args.chunk, reset_num_timesteps=False)
        extra += args.chunk
        st = evaluate_policy_at(model, eval_env, ctx, args.episodes, base_seed=123_000)

    converged = st["mean_return"] >= args.threshold
    model.save(os.path.join(args.run, "model"))
    if meta["algo"] == "sac":
        model.save_replay_buffer(os.path.join(args.run, "replay_buffer.pkl"))
    meta["steps"] = meta["steps"] + extra
    meta["extra_steps_for_convergence"] = extra
    meta["converged"] = bool(converged)
    meta["final_eval"] = st
    with open(os.path.join(args.run, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[converge] {args.run}: +{extra} steps -> {st['mean_return']:.1f} "
          f"(converged={converged})")


if __name__ == "__main__":
    main()
