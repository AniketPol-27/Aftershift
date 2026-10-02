"""Self-ID-MPC arm: detect -> identify -> adapt, without retraining.

Controller stack:
  * LQR gains from a numerically linearized cart-pole model at the current parameter
    estimate (DARE solved by value iteration -- no scipy dependency). NOTE: B is
    per-unit-ACTION, so K maps state -> action directly (units bug caught by validate).
  * Parameter identifier with two interchangeable implementations:
      - GridIdentifier (default): multi-hypothesis batch identification. A bank of
        candidate pole lengths is scored by Gaussian one-step prediction likelihood
        over a sliding window of recent transitions; the point estimate switches only
        after a hysteresis counter confirms a new winner. Robust to large initial
        parameter error (no linearization), unlike the EKF.
      - PoleLengthEKF: 5-state EKF over [x, x_dot, theta, theta_dot, l] (kept as an
        ablation; exhibits a certainty-equivalence stall on large shifts -- see
        the project research log).
  * Deployment monitor: nominal-physics residual CUSUM (same as the RL experiments).
  * On detection: clear the identifier window (pre-shift data is from the old
    dynamics), switch to certainty-equivalent adaptive LQR, and inject a scheduled
    sinusoidal dither (persistent excitation) while identifying.

Experiment protocol per (shift, seed):
  1. ONE adaptation episode with the mid-episode shift (t_shift=100): records
     detection time, identification time (l_hat within 5% of true, sustained 50
     steps), and the episode return.
  2. 10 frozen evaluation episodes at the shifted context (l_hat and gains fixed):
     post-adaptation performance.
Comparison targets: nominal-gain LQR (deployed, no adaptation) and oracle LQR
(gains computed with the true shifted parameter).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque

import numpy as np

from .detectors import CUSUMDetector, CartPolePhysicsPredictor, calibrate_cusum, residual_norm
from .envs import CARTPOLE_NOMINAL, ContextCartPole
from .shifts import ShiftScheduler, step_events

L_BOUNDS = (0.2, 3.0)
DITHER_PERIOD = 20
DITHER_MAX = 0.25
DITHER_GAIN = 0.5          # only used by the EKF variant (P-scaled dither)
# Scheduled probe (persistent-excitation schedule), steps-after-detection -> amplitude
DITHER_SCHEDULE = ((200, 0.15), (400, 0.05))
IDENTIFY_TOL = 0.05        # |l_hat - l_true| <= 5% of l_true
IDENTIFY_HOLD = 50         # ... sustained for 50 steps
ORACLE_DETECT_T = 100      # detection-ablation horizon (matches t_shift)

# Phase structure (fix for the transient-chasing failure mode):
#   NOMINAL -> (detection) -> PROBE -> (PROBE_LEN steps or episode end) -> ADAPT
# During PROBE the controller keeps NOMINAL gains and injects strong two-tone dither;
# the identifier only collects data. Gains change only in ADAPT, from batch-committed
# estimates. The estimate is finalized even if the episode dies during the probe.
PROBE_LEN = 200
PROBE_DITHER = 0.20
ADAPT_DITHER_SCHEDULE = ((100, 0.05),)   # mild refinement dither after committing

# EKF-specific constants (ablation variant)
L_PROCESS_VAR = 1e-7
L_VAR_FLOOR = 1e-3
L_INFLATE_ON_DETECT = 0.6


# ---------------- dynamics (shared with env, exact discrete step) ----------------

def cartpole_step(state: np.ndarray, a: float, ctx: dict) -> np.ndarray:
    """Exact (noiseless) one-step map used by both env and estimators."""
    x, x_dot, th, th_dot = state
    a = float(np.clip(a, -1.0, 1.0))
    force = a * ctx["force_mag"] - ctx["cart_friction"] * x_dot
    ct, st = np.cos(th), np.sin(th)
    mc, mp, l, g = ctx["masscart"], ctx["masspole"], ctx["pole_len"], ctx["gravity"]
    total = mc + mp
    mpl = mp * l
    temp = (force + mpl * th_dot ** 2 * st) / total
    th_acc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct ** 2 / total))
    x_acc = temp - mpl * th_acc * ct / total
    dt = 0.02
    return np.array([x + dt * x_dot, x_dot + dt * x_acc,
                     th + dt * th_dot, th_dot + dt * th_acc])


def cartpole_step_vec(states: np.ndarray, a: float, base_ctx: dict, l_arr: np.ndarray):
    """Vectorized over candidate l values. states: (K,4); l_arr: (K,)."""
    a = float(np.clip(a, -1.0, 1.0))
    x, x_dot, th, th_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    force = a * base_ctx["force_mag"] - base_ctx["cart_friction"] * x_dot
    ct, st = np.cos(th), np.sin(th)
    mc, mp, g = base_ctx["masscart"], base_ctx["masspole"], base_ctx["gravity"]
    total = mc + mp
    mpl = mp * l_arr
    temp = (force + mpl * th_dot ** 2 * st) / total
    th_acc = (g * st - ct * temp) / (l_arr * (4.0 / 3.0 - mp * ct ** 2 / total))
    x_acc = temp - mpl * th_acc * ct / total
    return np.stack([x + 0.02 * x_dot, x_dot + 0.02 * x_acc,
                     th + 0.02 * th_dot, th_dot + 0.02 * th_acc], axis=1)


# ---------------- LQR via numeric linearization + DARE value iteration ----------------

def linearize(ctx: dict):
    """Discrete A, B about upright equilibrium via central differences."""
    x0 = np.zeros(4)
    A = np.zeros((4, 4))
    for i in range(4):
        e = np.zeros(4); e[i] = 1e-6
        A[:, i] = (cartpole_step(x0 + e, 0.0, ctx) - cartpole_step(x0 - e, 0.0, ctx)) / 2e-6
    e = 1e-6
    B = (cartpole_step(x0, e, ctx) - cartpole_step(x0, -e, ctx)).reshape(4, 1) / (2e-6)
    return A, B


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, iters: int = 2000):
    """DARE by value iteration; returns gain K with u = -K x."""
    P = Q.copy()
    for _ in range(iters):
        BtPB = R + B.T @ P @ B
        K = np.linalg.solve(BtPB, B.T @ P @ A)
        Pn = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.max(np.abs(Pn - P)) < 1e-10:
            P = Pn
            break
        P = Pn
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


class LQRController:
    def __init__(self, ctx: dict, Q=None, R=None):
        self.ctx = dict(ctx)
        Q = np.diag([1.0, 0.5, 12.0, 0.5]) if Q is None else Q
        R = np.array([[0.1]]) if R is None else R
        A, B = linearize(self.ctx)
        self.K = dlqr(A, B, Q, R)

    def act(self, obs: np.ndarray) -> float:
        # B in linearize() is per-unit-ACTION, so K maps state -> action directly.
        a = float(-(self.K @ np.asarray(obs))[0])
        return float(np.clip(a, -1.0, 1.0))


# ---------------- identifier 1: multi-hypothesis grid (default) ----------------

class GridIdentifier:
    """Bank of candidate l values scored by one-step prediction likelihood."""

    def __init__(self, ctx: dict, obs_noise, window: int = 10_000, n_grid: int = 85,
                 hysteresis: int = 15):
        self.base = {k: v for k, v in ctx.items() if k != "pole_len"}
        self.l_grid = np.linspace(L_BOUNDS[0], L_BOUNDS[1], n_grid)
        self.l_hat = float(ctx["pole_len"])
        self.sigma = np.asarray(obs_noise, dtype=float)
        self.window = deque(maxlen=window)
        self.hysteresis = hysteresis
        self._pending = None
        self._pending_count = 0

    def update(self, obs_prev, a: float, obs_new) -> None:
        self.window.append((np.asarray(obs_prev, dtype=float), float(a),
                            np.asarray(obs_new, dtype=float)))

    def clear(self) -> None:
        self.window.clear()

    def _score(self):
        """Negative log-likelihood of the window under each candidate l."""
        scores = np.zeros(len(self.l_grid))
        for obs_prev, a, obs_new in self.window:
            preds = cartpole_step_vec(np.tile(obs_prev, (len(self.l_grid), 1)),
                                      a, self.base, self.l_grid)
            r = (obs_new[None, :] - preds) / self.sigma[None, :]
            scores += np.sum(r ** 2, axis=1)
        return scores

    def refresh(self) -> float:
        """No-op: the grid variant commits estimates only via decide()."""
        return self.l_hat

    def decide(self, min_items: int = 30) -> float:
        """Batch commit: set l_hat to the window maximum-likelihood candidate.

        Committing only on a full window (not chasing transients step-by-step) is the
        fix for the 'falling long pole looks like a fast short pole' trap: short
        divergence windows are maximized by small-l candidates, so step-wise
        switching death-spirals (see the project research log).
        """
        if len(self.window) < min_items:
            return self.l_hat
        self.l_hat = float(self.l_grid[np.argmin(self._score())])
        return self.l_hat

    @property
    def l_std(self) -> float:
        return 0.0  # grid variant uses the scheduled dither only


# ---------------- identifier 2: EKF (ablation) ----------------

class PoleLengthEKF:
    def __init__(self, ctx: dict, obs_noise: np.ndarray, l0: float | None = None):
        self.ctx = dict(ctx)
        self.base = {k: v for k, v in self.ctx.items() if k != "pole_len"}
        self.l = float(self.ctx["pole_len"]) if l0 is None else l0
        self.y = np.array([0.0, 0.0, 0.0, 0.0, self.l])
        self.P = np.diag([1e-3, 1e-2, 1e-3, 1e-2, 1e-1])
        self.R_obs = np.diag(np.asarray(obs_noise) ** 2)
        self.Q_proc = np.diag([1e-7, 1e-5, 1e-7, 1e-5, L_PROCESS_VAR])
        self.H = np.zeros((4, 5)); self.H[:, :4] = np.eye(4)

    def _f(self, y: np.ndarray, a: float) -> np.ndarray:
        ctx = dict(self.base); ctx["pole_len"] = float(y[4])
        nxt = cartpole_step(y[:4], a, ctx)
        return np.concatenate([nxt, [y[4]]])

    def _F(self, y: np.ndarray, a: float) -> np.ndarray:
        F = np.eye(5)
        for i in range(5):
            e = np.zeros(5); e[i] = 1e-6
            F[:, i] = (self._f(y + e, a) - self._f(y - e, a)) / 2e-6
        return F

    def decide(self, min_items: int = 0):
        return self.l_hat

    def predict_update(self, obs_prev: np.ndarray, a: float, obs_new: np.ndarray) -> None:
        y = self.y.copy()
        y[:4] = np.asarray(obs_prev, dtype=float)
        y_pred = self._f(y, a)
        F = self._F(y, a)
        P = F @ self.P @ F.T + self.Q_proc
        P = 0.5 * (P + P.T)
        v = np.asarray(obs_new, dtype=float) - y_pred[:4]
        S = self.H @ P @ self.H.T + self.R_obs
        K = P @ self.H.T @ np.linalg.inv(S)
        self.y = y_pred + K @ v
        self.y[4] = float(np.clip(self.y[4], *L_BOUNDS))
        A_post = np.eye(5) - K @ self.H
        self.P = A_post @ P @ A_post.T + K @ self.R_obs @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P[4, 4] = max(self.P[4, 4], L_VAR_FLOOR)

    # common API with GridIdentifier
    def update(self, obs_prev, a, obs_new):
        self.predict_update(obs_prev, a, obs_new)

    def clear(self):
        pass

    def refresh(self):
        return self.l_hat

    @property
    def l_hat(self) -> float:
        return float(self.y[4])

    @property
    def l_std(self) -> float:
        return float(np.sqrt(max(self.P[4, 4], 0.0)))

    def inflate_l(self, var: float = L_INFLATE_ON_DETECT) -> None:
        self.P[4, 4] = max(self.P[4, 4], var)


# ---------------- Self-ID controller (detect -> identify -> adapt) ----------------

class SelfIDController:
    """Phased detect-identify-adapt controller (see module docstring).

    Phases: NOMINAL -> PROBE (nominal gains + strong dither, estimation only)
            -> ADAPT (batch-committed estimate drives certainty-equivalent LQR).
    """

    def __init__(self, nominal_ctx: dict, obs_noise, cusum: CUSUMDetector,
                 dither: bool = True, oracle_detect: bool = False,
                 identifier: str = "grid"):
        self.nominal_ctx = dict(nominal_ctx)
        self.lqr_nominal = LQRController(nominal_ctx)
        if identifier == "grid":
            self.ident = GridIdentifier(nominal_ctx, obs_noise)
        elif identifier == "ekf":
            self.ident = PoleLengthEKF(nominal_ctx, obs_noise)
        else:
            raise ValueError(identifier)
        self.identifier_kind = identifier
        self.cusum = cusum
        self.predictor = CartPolePhysicsPredictor(nominal_ctx)
        self.dither_on = dither
        self.oracle_detect = oracle_detect
        self.phase = "NOMINAL"
        self.t_detect = None
        self.t_decide = None
        self._lqr_adapt: LQRController | None = None
        self._lqr_adapt_l = None
        self._t = 0            # per-sub-episode counter (dither signal phase)
        self.steps = 0         # monotonic across sub-episodes (all scheduling)

    # ---- gains ----
    def _adaptive_action(self, obs: np.ndarray) -> float:
        l_hat = self.ident.l_hat
        if self._lqr_adapt is None or abs(l_hat - self._lqr_adapt_l) > 0.02 * max(self._lqr_adapt_l, 1e-6):
            ctx = dict(self.nominal_ctx); ctx["pole_len"] = l_hat
            self._lqr_adapt = LQRController(ctx)
            self._lqr_adapt_l = l_hat
        return self._lqr_adapt.act(obs)

    def act(self, obs: np.ndarray) -> float:
        if self.phase == "NOMINAL":
            return self.lqr_nominal.act(obs)
        if self.phase == "PROBE":
            a = self.lqr_nominal.act(obs)  # keep nominal gains; do not chase transients
            if self.dither_on:
                d = (PROBE_DITHER * np.sin(2 * np.pi * self._t / 20)
                     + 0.5 * PROBE_DITHER * np.sin(2 * np.pi * self._t / 7))
                a = float(np.clip(a + d, -1.0, 1.0))
            return a
        # ADAPT
        a = self._adaptive_action(obs)
        if self.dither_on and self.t_decide is not None:
            t_since = self._t - self.t_decide
            sched = 0.0
            for horizon, amp in ADAPT_DITHER_SCHEDULE:
                if t_since < horizon:
                    sched = amp
                    break
            if sched > 0:
                a = float(np.clip(a + sched * np.sin(2 * np.pi * self._t / 20), -1.0, 1.0))
        return a

    # ---- estimation + monitoring ----
    def observe(self, obs_prev, a_prev: float, obs_new) -> bool:
        """Estimator + monitor update; returns True on the detection step (latched)."""
        self._t += 1
        self.steps += 1
        self.ident.update(obs_prev, a_prev, obs_new)
        r = residual_norm(self.predictor.predict(np.asarray(obs_prev), np.array([a_prev])), obs_new)
        fired = self.cusum.update(r)
        if self.oracle_detect:
            fired = self._t >= ORACLE_DETECT_T
        if self.phase == "NOMINAL" and fired:
            self.phase = "PROBE"
            self.t_detect = self.steps
            self.ident.clear()  # pre-shift data belongs to the old dynamics
            if isinstance(self.ident, PoleLengthEKF):
                self.ident.inflate_l()
            return True
        if self.phase == "PROBE" and (self.steps - self.t_detect) >= PROBE_LEN:
            self.ident.decide()
            self.phase = "ADAPT"
            self.t_decide = self.steps
        elif self.phase == "ADAPT" and (self.steps - self.t_decide) >= 200:
            self.ident.decide()  # periodic batch re-commit on the growing window
            self.t_decide = self.steps
        return False

    def finalize(self):
        """Commit a final estimate whenever the window has enough data.

        Must run regardless of phase: if sub-episodes die faster than PROBE_LEN,
        the commit would otherwise never happen and the controller death-loops on
        stale gains (bug caught by the 3-seed CUSUM grid).
        """
        self.ident.decide(min_items=20)
        self.phase = "ADAPT"
        self.t_decide = self.steps

    def reset(self, carry_identifier: bool = False):
        """Full reset, or a soft reset that keeps the committed estimate.

        carry_identifier=True is used between sub-episodes of one adaptation run:
        the plant stays shifted, the identifier keeps its window and committed
        l_hat, and the controller resumes in the ADAPT phase.
        """
        if not carry_identifier:
            if isinstance(self.ident, GridIdentifier):
                self.ident = GridIdentifier(self.nominal_ctx, self.ident.sigma)
            else:
                obs_noise = np.sqrt(np.diag(self.ident.R_obs))
                self.ident = PoleLengthEKF(self.nominal_ctx, obs_noise)
            self.cusum.reset()
            self.phase = "NOMINAL"
            self.t_detect = None
            self.t_decide = None
            self._lqr_adapt = None
            self._lqr_adapt_l = None
            self._t = 0
            self.steps = 0
            return
        # soft reset: keep ident (window + l_hat), phase, timing records, and
        # the monotonic step counter (scheduling must survive sub-episode deaths)
        self.cusum.reset()
        self._lqr_adapt = None
        self._lqr_adapt_l = None
        self._t = 0


# ---------------- experiments ----------------

def rollout(env: ContextCartPole, controller, l_true: float, seed: int, t_shift: int = 100,
            budget: int = 1000):
    """Adaptation run: sub-episodes under a shifted plant, within an interaction budget.

    Sub-episode 0 starts at nominal context with the shift at `t_shift`. If the
    controller fails (pole falls) it keeps trying (soft reset; identifier carried
    over; plant shifted from step 0) until it survives a full episode in the ADAPT
    phase or the post-shift interaction budget is exhausted. Mirrors the
    budget-matched comparison with fine-tuning: recovery cost = post-shift steps.
    """
    controller.reset()
    total_post = 0
    sub = 0
    t_detect = None
    t_identify = None
    identify_run = 0
    l_trace = []
    total_R = 0.0
    last_R = 0.0
    last_len = 0
    recovered = False
    any_failed = False
    while total_post < budget and not recovered:
        obs, _ = env.reset(seed=seed * 1000 + sub)
        env.set_context(**CARTPOLE_NOMINAL)
        sched = ShiftScheduler(step_events(t_shift if sub == 0 else 0, pole_len=l_true))
        sched.reset()
        if sub > 0:
            controller.reset(carry_identifier=True)
        obs_prev, a_prev = None, None
        done, t, R = False, 0, 0.0
        while not done:
            sched.apply(env, t)
            if obs_prev is not None:
                controller.observe(obs_prev, a_prev, obs)
                if t_detect is None and controller.t_detect is not None:
                    t_detect = controller.t_detect
            a = controller.act(obs)
            obs_prev, a_prev = obs, a
            obs, r, term, trunc, _ = env.step(np.array([a]))
            R += float(r)
            t += 1
            l_trace.append(float(controller.ident.l_hat))
            if (sub > 0) or (t > t_shift):
                total_post += 1
                if controller.phase != "NOMINAL":
                    if abs(controller.ident.l_hat - l_true) <= IDENTIFY_TOL * l_true:
                        identify_run += 1
                        if identify_run >= IDENTIFY_HOLD and t_identify is None:
                            t_identify = total_post
                    else:
                        identify_run = 0
                if total_post >= budget:
                    done = True
                    break
            done = done or term or trunc
        any_failed = any_failed or term
        total_R += R
        last_R, last_len = R, t
        if trunc and controller.phase == "ADAPT" and controller.t_decide is not None:
            recovered = True
        controller.finalize()  # commits a final estimate if the episode died in PROBE
        sub += 1
    if (t_identify is None and controller.phase != "NOMINAL"
            and abs(controller.ident.l_hat - l_true) <= IDENTIFY_TOL * l_true):
        t_identify = total_post
    return dict(return_=total_R, last_return=last_R, length=last_len,
                failed=any_failed and not recovered,
                t_detect=t_detect, t_identify=t_identify,
                l_hat_final=float(controller.ident.l_hat), l_trace=l_trace,
                recovered=recovered, interactions_post=total_post, sub_episodes=sub)


def eval_frozen(env: ContextCartPole, controller, l_true: float, seed0: int, episodes: int = 10):
    """Post-adaptation evaluation: frozen controller at the shifted context."""
    rets = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        env.set_context(pole_len=l_true)
        done, R = False, 0.0
        while not done:
            a = controller.act(obs)
            obs, r, term, trunc, _ = env.step(np.array([a]))
            R += float(r)
            done = term or trunc
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))


def make_cusum(env: ContextCartPole, lqr: LQRController, predictor, episodes: int = 50,
               base_seed: int = 600_000):
    """Calibrate the residual CUSUM on nominal LQR episodes (5% in-sample FA)."""
    streams = []
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        env.set_context(**CARTPOLE_NOMINAL)
        res, done, prev, a_prev = [], False, None, None
        while not done:
            a = lqr.act(obs)
            if prev is not None:
                res.append(residual_norm(predictor.predict(prev, np.array([a_prev])), obs))
            prev, a_prev = obs, a
            obs, r, term, trunc, _ = env.step(np.array([a]))
            done = term or trunc
        streams.append(res)
    nu, k, h, fpr = calibrate_cusum(streams, target_fpr=0.05)
    return CUSUMDetector(nu, k, h), (nu, k, h, fpr)


def run_grid(shift_ls, seeds, dither: bool, oracle_detect: bool, out_csv: str,
             identifier: str = "grid", t_shift: int = 100):
    env = ContextCartPole()
    predictor = CartPolePhysicsPredictor()
    lqr_nom = LQRController(CARTPOLE_NOMINAL)
    cusum, cal = make_cusum(env, lqr_nom, predictor)
    print(f"[selfid] CUSUM calibrated: nu={cal[0]:.4f} k={cal[1]:.4f} h={cal[2]:.3f} FA={cal[3]:.2f}")

    from .envs import PILOT_OBS_NOISE
    done = set()
    if os.path.exists(out_csv):
        try:
            import pandas as _pd
            _old = _pd.read_csv(out_csv)
            done = set(zip(_old["l_true"], _old["seed"], _old["identifier"],
                           _old["dither"].astype(bool), _old["oracle_detect"].astype(bool)))
        except Exception:
            done = set()
    rows = []
    for l_true in shift_ls:
        m_nom, _ = eval_frozen(env, lqr_nom, l_true, 3_000_000)
        ctx_o = dict(CARTPOLE_NOMINAL); ctx_o["pole_len"] = l_true
        m_orc, _ = eval_frozen(env, LQRController(ctx_o), l_true, 3_100_000)
        print(f"[selfid] l={l_true}: nominal-LQR {m_nom:.0f} | oracle-LQR {m_orc:.0f}")
        for seed in seeds:
            if (l_true, seed, identifier, bool(dither), bool(oracle_detect)) in done:
                print(f"[selfid] skip l={l_true} seed={seed} (already in {out_csv})")
                continue
            sid = SelfIDController(CARTPOLE_NOMINAL, PILOT_OBS_NOISE,
                                   CUSUMDetector(*cal[:3]), dither=dither,
                                   oracle_detect=oracle_detect, identifier=identifier)
            ad = rollout(env, sid, l_true, seed=1_000_000 + seed, t_shift=t_shift)
            sid.dither_on = False  # freeze for evaluation
            m_post, s_post = eval_frozen(env, sid, l_true, 2_000_000 + 100 * seed)
            rows.append(dict(
                l_true=l_true, seed=seed, identifier=identifier, dither=dither,
                oracle_detect=oracle_detect,
                adapt_total_return=ad["return_"], adapt_last_return=ad["last_return"],
                recovered=ad["recovered"], interactions_post=ad["interactions_post"],
                sub_episodes=ad["sub_episodes"],
                t_detect=ad["t_detect"], t_identify=ad["t_identify"],
                l_hat_final=ad["l_hat_final"],
                nominal_lqr_return=m_nom, oracle_lqr_return=m_orc,
                post_adapt_return=m_post, post_adapt_std=s_post,
                post_adapt_norm=m_post / 500.0,
                oracle_norm=m_orc / 500.0,
            ))
            print(f"[selfid] l={l_true} seed={seed}: recovered={ad['recovered']} "
                  f"post_steps={ad['interactions_post']} (subs={ad['sub_episodes']}) "
                  f"t_det={ad['t_detect']} t_ident={ad['t_identify']} "
                  f"l_hat={ad['l_hat_final']:.3f} post={m_post:.0f} (oracle {m_orc:.0f})")

    if not rows:
        print("[selfid] nothing new to run (all rows present)")
        import pandas as pd
        df = pd.read_csv(out_csv)
        df = df[(df["identifier"] == identifier) & (df["dither"] == bool(dither))
                & (df["oracle_detect"] == bool(oracle_detect))]
    else:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        new = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        import pandas as pd
        df = pd.read_csv(out_csv)
        df = df[(df["identifier"] == identifier) & (df["dither"] == bool(dither))
                & (df["oracle_detect"] == bool(oracle_detect))]
    summary = {}
    for l_true, g in df.groupby("l_true"):
        summary[str(l_true)] = dict(
            n=len(g),
            recovery_rate=float(g["recovered"].mean()),
            median_interactions_post=float(g["interactions_post"].median()),
            post_adapt_norm_mean=float(g["post_adapt_norm"].mean()),
            oracle_norm_mean=float(g["oracle_norm"].mean()),
            nominal_norm_mean=float(g["nominal_lqr_return"].mean() / 500.0),
            identify_rate=float(g["t_identify"].notna().mean()),
            median_t_identify=float(g["t_identify"].dropna().median()) if g["t_identify"].notna().any() else None,
            median_detect=float(g["t_detect"].dropna().median()) if g["t_detect"].notna().any() else None,
            median_l_hat_error=float((g["l_hat_final"] - l_true).abs().median()),
        )
    with open(out_csv.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def validate():
    """Fast correctness checks (no full grid)."""
    env = ContextCartPole()
    lqr = LQRController(CARTPOLE_NOMINAL)
    m, s = eval_frozen(env, lqr, 0.5, 4242, episodes=5)
    print(f"[validate] nominal LQR: {m:.0f} +- {s:.0f} (expect ~500)")
    assert m > 490, "nominal LQR must solve the nominal task"
    from .envs import PILOT_OBS_NOISE
    cusum, _ = make_cusum(env, lqr, CartPolePhysicsPredictor(), episodes=20)
    for identifier in ("grid", "ekf"):
        sid = SelfIDController(CARTPOLE_NOMINAL, PILOT_OBS_NOISE, cusum,
                               oracle_detect=True, identifier=identifier)
        for l_true in (1.0, 2.0, 2.5):
            ad = rollout(env, sid, l_true, seed=77)
            print(f"[validate] {identifier} l={l_true}: l_hat_final={ad['l_hat_final']:.3f} "
                  f"t_ident={ad['t_identify']} recovered={ad['recovered']} "
                  f"post_steps={ad['interactions_post']} last_ret={ad['last_return']:.0f}")
    # grid identifier must nail l on a dithered episode
    print("[validate] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--shifts", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--no-dither", dest="dither", action="store_false")
    ap.add_argument("--oracle-detect", action="store_true")
    ap.add_argument("--identifier", choices=["grid", "ekf"], default="grid")
    ap.add_argument("--out", default="results/selfid.csv")
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        run_grid([float(x) for x in args.shifts.split(",")],
                 [int(x) for x in args.seeds.split(",")],
                 dither=args.dither, oracle_detect=args.oracle_detect,
                 identifier=args.identifier, out_csv=args.out)


if __name__ == "__main__":
    main()
