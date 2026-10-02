"""Context-parameterized environments for AFTERSHIFT.

Physics follows the standard Gymnasium CartPole / Pendulum formulations, extended with:
  * explicit context parameters (masses, lengths, actuator strength, friction, gravity)
  * viscous friction terms (absent from the classic envs)
  * runtime context mutation via `set_context(**params)` for mid-episode shifts

Conventions
-----------
CartPole: `pole_len` is the pivot-to-center-of-mass distance (Gymnasium `length`).
Action is continuous force in [-1, 1], scaled by `force_mag` (decision D1 in the project research log:
continuous actions make actuator-strength degradation a smooth, graded phenomenon and
match the continuous-control framing of the protocol).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

CARTPOLE_NOMINAL = {
    "masscart": 1.0,
    "masspole": 0.1,
    "pole_len": 0.5,
    "force_mag": 10.0,
    "cart_friction": 0.0,
    "gravity": 9.8,
}

# Pilot sensor noise (per-dim std on observations). Deviation D6 in the project research log:
# without noise, the nominal-model residual is identically zero and detection is
# degenerate; this makes the monitor problem non-trivial and realistic.
PILOT_OBS_NOISE = (0.005, 0.02, 0.005, 0.02)

PENDULUM_NOMINAL = {
    "mass": 1.0,
    "length": 1.0,
    "gravity": 9.8,
    "max_torque": 2.0,
    "damping": 0.0,
}


class ContextCartPole(gym.Env):
    """Continuous-force cart-pole with explicit, mutable physical context.

    Effective force = action * force_mag - cart_friction * x_dot  (viscous).
    Reward: +1 per surviving step. Episode ends at 500 steps, |x| > 2.4, or |theta| > 0.2095.
    """

    metadata = {"render_modes": []}

    def __init__(self, context: dict | None = None, max_steps: int = 500,
                 theta_lim: float = 0.2095, x_lim: float = 2.4, dt: float = 0.02,
                 obs_noise=None):
        super().__init__()
        self.nominal = dict(CARTPOLE_NOMINAL)
        self.ctx = dict(self.nominal)
        if context:
            self.ctx.update(context)
        self.max_steps = max_steps
        self.theta_lim = theta_lim
        self.x_lim = x_lim
        self.dt = dt
        if obs_noise is None:
            obs_noise = PILOT_OBS_NOISE
        self.obs_noise = np.asarray(obs_noise, dtype=float)
        if self.obs_noise.shape != (4,):
            raise ValueError("obs_noise must have shape (4,)")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        self.state: np.ndarray | None = None
        self._t = 0

    def _obs(self) -> np.ndarray:
        """Noisy observation of the internal state (policy and monitor both see this)."""
        return self.state + self.np_random.normal(0.0, self.obs_noise)

    # ---- context API (the core extension over standard envs) ----
    def get_context(self) -> dict:
        return dict(self.ctx)

    def set_context(self, **params) -> None:
        self.ctx.update(params)

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self._t = 0
        return self.state.copy(), {}

    def step(self, action):
        assert self.state is not None, "call reset() first"
        x, x_dot, th, th_dot = self.state
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        force = a * self.ctx["force_mag"] - self.ctx["cart_friction"] * x_dot
        ct, st = np.cos(th), np.sin(th)
        mc = self.ctx["masscart"]
        mp = self.ctx["masspole"]
        l = self.ctx["pole_len"]
        g = self.ctx["gravity"]
        total = mc + mp
        mpl = mp * l
        temp = (force + mpl * th_dot ** 2 * st) / total
        th_acc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct ** 2 / total))
        x_acc = temp - mpl * th_acc * ct / total
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        th = th + self.dt * th_dot
        th_dot = th_dot + self.dt * th_acc
        self.state = np.array([x, x_dot, th, th_dot], dtype=np.float64)
        terminated = bool(abs(x) > self.x_lim or abs(th) > self.theta_lim)
        truncated = bool(self._t + 1 >= self.max_steps)
        self._t += 1
        return self._obs(), 1.0, terminated, truncated, {"context": self.get_context()}


class ContextPendulum(gym.Env):
    """Continuous-torque pendulum (swing-up + balance) with mutable context.

    obs = [cos(theta), sin(theta), theta_dot]; action in [-1, 1] scaled by max_torque.
    Total torque = action * max_torque - damping * theta_dot. Reward = -(cost).
    """

    metadata = {"render_modes": []}

    def __init__(self, context: dict | None = None, max_steps: int = 200, dt: float = 0.05):
        super().__init__()
        self.nominal = dict(PENDULUM_NOMINAL)
        self.ctx = dict(self.nominal)
        if context:
            self.ctx.update(context)
        self.max_steps = max_steps
        self.dt = dt
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
        self.state: np.ndarray | None = None  # [theta, theta_dot]
        self._t = 0

    def get_context(self) -> dict:
        return dict(self.ctx)

    def set_context(self, **params) -> None:
        self.ctx.update(params)

    def _obs(self) -> np.ndarray:
        th, th_dot = self.state
        return np.array([np.cos(th), np.sin(th), th_dot], dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        th = self.np_random.uniform(low=-np.pi, high=np.pi)
        th_dot = self.np_random.uniform(low=-1.0, high=1.0)
        self.state = np.array([th, th_dot], dtype=np.float64)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        th, th_dot = self.state
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        u = a * self.ctx["max_torque"] - self.ctx["damping"] * th_dot
        m, l, g = self.ctx["mass"], self.ctx["length"], self.ctx["gravity"]
        new_th_dot = th_dot + (3.0 * g / (2.0 * l) * np.sin(th) + 3.0 * u / (m * l ** 2)) * self.dt
        new_th = th + self.dt * th_dot
        new_th = (new_th + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([new_th, new_th_dot], dtype=np.float64)
        self._t += 1
        truncated = bool(self._t >= self.max_steps)
        cost = float(np.clip(new_th, -np.pi, np.pi) ** 2 + 0.1 * new_th_dot ** 2 + 0.001 * u ** 2)
        return self._obs(), -cost, False, truncated, {"context": self.get_context()}
