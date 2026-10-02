"""Train AFTERSHIFT baselines: SAC / PPO at nominal context, DR-SAC, shifted oracle."""
from __future__ import annotations

import argparse
import json
import os
import time

import gymnasium as gym
from stable_baselines3 import SAC, PPO

from .envs import CARTPOLE_NOMINAL, ContextCartPole, ContextPendulum
from .evaluate import evaluate_policy_at

# Domain-randomization ranges (documented decision D4 in the project research log)
DR_RANGES = {
    "masscart": (0.5, 2.0),
    "masspole": (0.05, 0.2),
    "pole_len": (0.25, 1.0),
    "force_mag": (5.0, 15.0),
    "cart_friction": (0.0, 1.0),
}


class ContextRandomizer(gym.Wrapper):
    """Resamples the context from `ranges` at every reset (domain randomization)."""

    def __init__(self, env, ranges):
        super().__init__(env)
        self.ranges = ranges

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ctx = {k: float(self.np_random.uniform(lo, hi)) for k, (lo, hi) in self.ranges.items()}
        self.env.set_context(**ctx)
        return obs, info



def parse_context(s: str) -> dict:
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


def build_env(env_name: str, context: dict, dr: bool):
    cls = ContextCartPole if env_name == "cartpole" else ContextPendulum
    env = cls(context=context)
    if dr:
        env = ContextRandomizer(env, DR_RANGES)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "ppo"], required=True)
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--context", default="{}", help="JSON context overrides (e.g. oracle shift)")
    ap.add_argument("--dr", action="store_true", help="train with domain randomization")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    context = parse_context(args.context)
    env = build_env(args.env, context, args.dr)

    if args.algo == "sac":
        model = SAC("MlpPolicy", env, seed=args.seed, learning_rate=3e-4,
                    buffer_size=100_000, batch_size=128, learning_starts=2000,
                    train_freq=1, gradient_steps=1,
                    policy_kwargs=dict(net_arch=[128, 128]), verbose=0)
    else:
        model = PPO("MlpPolicy", env, seed=args.seed, learning_rate=3e-4,
                    n_steps=1024, batch_size=256, n_epochs=10,
                    policy_kwargs=dict(net_arch=[64, 64]), verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=args.steps)
    wall = time.time() - t0

    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model.save(os.path.join(run_dir, "model"))
    if args.algo == "sac":
        model.save_replay_buffer(os.path.join(run_dir, "replay_buffer.pkl"))

    eval_ctx = dict(CARTPOLE_NOMINAL)
    eval_ctx.update(context)
    stats = evaluate_policy_at(model, build_env(args.env, context, dr=False), eval_ctx,
                               episodes=10, base_seed=123_000)

    meta = dict(name=args.name, algo=args.algo, env=args.env, context=context, dr=bool(args.dr),
                steps=args.steps, seed=args.seed, wall_time_s=round(wall, 1),
                fps=round(args.steps / wall, 1), final_eval=stats,
                dr_ranges={k: list(v) for k, v in DR_RANGES.items()} if args.dr else None,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] {args.name}: return {stats['mean_return']:.1f} +- {stats['std_return']:.1f} "
          f"| {args.steps} steps in {wall:.0f}s ({args.steps / wall:.0f} fps)")


if __name__ == "__main__":
    main()
