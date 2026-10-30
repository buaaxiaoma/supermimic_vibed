"""Deterministic rollouts and xi-sensitivity analysis, shared by the simulator and the world model.

Both rollouts use the *same* analytic reward/termination of :mod:`supermimic.mdp` and the hard
termination threshold (never the soft one used in imagination), so that their returns are
comparable. No critic bootstrap is applied anywhere here.

``xi`` layout (33): ``[contact friction mu, 29 joint zero offsets, 3 torso CoM offsets]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from .mdp import MDP, NUM_J, XI_DIM, normalize_xi
from .motion import MotionLib
from .robot import G1_JOINT_NAMES

if TYPE_CHECKING:  # the simulator pulls in Isaac Lab, which the world-model path must not require
    from .env import TrackingSim
    from .networks import Policy, WorldModel

XI_NAMES: list[str] = (
    ["friction"]
    + [f"joint_offset:{name}" for name in G1_JOINT_NAMES]
    + ["com:x", "com:y", "com:z"]
)
assert len(XI_NAMES) == XI_DIM == 33, f"xi layout drifted: {len(XI_NAMES)} names for XI_DIM={XI_DIM}"


@dataclass
class Rollout:
    """Trajectory over ``steps`` control steps; every tensor is time-major."""

    S: tuple[torch.Tensor, ...]  # 6 tensors of shape (steps + 1, B, ...)
    a: torch.Tensor  # (steps, B, 29)
    r: torch.Tensor  # (steps, B), already masked by ``alive``
    alive: torch.Tensor  # (steps, B) bool, BEFORE the step's termination is applied
    term: torch.Tensor  # (steps, B) bool, first termination only


def set_xi(env: "TrackingSim", xi: torch.Tensor, *, update_com: bool = True) -> None:
    """Write a full xi vector into the simulator: contact friction, torso CoM, joint offsets.

    ``TrackingSim`` applies the joint offsets at use time (action target / observation), so only
    friction and CoM need pushing into the physics model.
    """
    xi = xi.to(env.device)
    env.xi[:] = xi
    env._apply_friction()
    if update_com:
        com = env.default_torso_com + xi[:, 30:33]
        env.robot.set_coms_index(coms=com.unsqueeze(1), body_ids=[env.torso_id])


def _discounted_return(r: torch.Tensor, alive: torch.Tensor, gamma: float) -> torch.Tensor:
    """Undiscounted-if-gamma-1 sum of masked rewards, shape (B,)."""
    return (r * (gamma ** torch.arange(r.shape[0], device=r.device, dtype=r.dtype).unsqueeze(-1))).sum(0)


def rollout_sim(
    env: "TrackingSim",
    pi: Optional["Policy"],
    mdp: MDP,
    motion: MotionLib,
    *,
    frames: torch.Tensor,
    xi: torch.Tensor,
    steps: int,
    goal_frames: int = 10,
    action_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    xi_obs: Optional[torch.Tensor] = None,
) -> Rollout:
    """Closed-loop rollout in the physics simulator from ``frames`` with a prescribed xi.

    ``xi_obs`` separates the observation channel from the physics: when given, the simulator
    keeps ``xi`` while the policy's observation and the reward reference use ``xi_obs``. This is
    how the "observation-only" control arm isolates a parameter change that the policy *sees*
    from one that actually changes the plant.
    """
    if pi is None and action_fn is None:
        raise ValueError("rollout_sim needs either a policy or an action_fn")
    dev = env.device
    ids = torch.arange(env.N, device=dev)
    xi_obs = xi if xi_obs is None else xi_obs
    hist_len = pi.hist if pi is not None else 1

    set_xi(env, xi)
    env.reset(ids, frames)
    env.refresh()

    S = env.state()
    states = [S]
    acts, rews, alives, terms = [], [], [], []
    alive = torch.ones(env.N, dtype=torch.bool, device=dev)
    hist = None
    for _ in range(steps):
        o = mdp.obs(S, env.a_prev, xi_obs, env.f)
        hist = o.unsqueeze(1).repeat(1, hist_len, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = action_fn(hist, env.f) if action_fn is not None else pi(hist, motion.goal(env.f, goal_frames))
        out = env.step(a)
        S = env.state()
        term = out["term"] & alive
        alives.append(alive.clone())
        terms.append(term)
        acts.append(a)
        rews.append(out["r"] * alive.to(out["r"].dtype))
        states.append(S)
        alive = alive & ~term
    return Rollout(
        S=tuple(torch.stack([s[i] for s in states], 0) for i in range(6)),
        a=torch.stack(acts, 0),
        r=torch.stack(rews, 0),
        alive=torch.stack(alives, 0),
        term=torch.stack(terms, 0),
    )


def rollout_wm(
    wm: "WorldModel",
    pi: Optional["Policy"],
    mdp: MDP,
    motion: MotionLib,
    *,
    S0: tuple[torch.Tensor, ...],
    xi: torch.Tensor,
    frames: torch.Tensor,
    steps: int,
    goal_frames: int = 10,
    action_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Rollout:
    """Closed-loop rollout in the world model, mirroring :func:`rollout_sim` step for step."""
    if pi is None and action_fn is None:
        raise ValueError("rollout_wm needs either a policy or an action_fn")
    dev = S0[0].device
    xi_n = normalize_xi(xi)
    hist_len = pi.hist if pi is not None else 1
    clip_end = motion.clip_end[motion.clip_id[frames]]

    S = S0
    f = frames.clone()
    a_prev = torch.zeros(S0[4].shape[0], NUM_J, device=dev)
    states = [S]
    acts, rews, alives, terms = [], [], [], []
    alive = torch.ones(S0[4].shape[0], dtype=torch.bool, device=dev)
    hist = None
    for _ in range(steps):
        o = mdp.obs(S, a_prev, xi, f)
        hist = o.unsqueeze(1).repeat(1, hist_len, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = action_fn(hist, f) if action_fn is not None else pi(hist, motion.goal(f, goal_frames))
        S1 = wm.step(S, a, xi_n)
        f1 = torch.minimum(f + 1, clip_end)
        r = mdp.reward(S1, f1, a, a_prev)
        term = mdp.termination(S1, f1) & alive
        alives.append(alive.clone())
        terms.append(term)
        acts.append(a)
        rews.append(r * alive.to(r.dtype))
        states.append(S1)
        alive = alive & ~term
        S, f, a_prev = S1, f1, a
    return Rollout(
        S=tuple(torch.stack([s[i] for s in states], 0) for i in range(6)),
        a=torch.stack(acts, 0),
        r=torch.stack(rews, 0),
        alive=torch.stack(alives, 0),
        term=torch.stack(terms, 0),
    )


def return_of(rollout: Rollout, gamma: float = 0.99) -> torch.Tensor:
    """Discounted, termination-masked return per batch element, shape (B,)."""
    return _discounted_return(rollout.r, rollout.alive, gamma)


def sensitivity_scan(
    env: "TrackingSim",
    wm: "WorldModel",
    pi: "Policy",
    mdp: MDP,
    motion: MotionLib,
    *,
    frames: torch.Tensor,
    xi_base: torch.Tensor,
    direction_idx: int,
    delta: float,
    steps: int,
    goal_frames: int = 10,
    gamma: float = 0.99,
) -> dict[str, torch.Tensor]:
    """Return the one-sided sensitivity of the return to ``xi[:, direction_idx]``.

    Returns ``dJ_sim`` and ``dJ_wm`` (physics *and* observation channel), plus ``obs_only``
    (observation channel alone: the plant keeps ``xi_base``). The physical-plant part is
    ``dJ_sim - obs_only``.
    """
    plus = xi_base.clone()
    minus = xi_base.clone()
    plus[:, direction_idx] += delta
    minus[:, direction_idx] -= delta

    kw = dict(frames=frames, steps=steps, goal_frames=goal_frames)
    j_sim_p = return_of(rollout_sim(env, pi, mdp, motion, xi=plus, **kw), gamma)
    j_sim_m = return_of(rollout_sim(env, pi, mdp, motion, xi=minus, **kw), gamma)
    j_obs_p = return_of(rollout_sim(env, pi, mdp, motion, xi=xi_base, xi_obs=plus, **kw), gamma)
    j_obs_m = return_of(rollout_sim(env, pi, mdp, motion, xi=xi_base, xi_obs=minus, **kw), gamma)

    S0 = tuple(x.clone() for x in motion.ref_state(frames))
    j_wm_p = return_of(rollout_wm(wm, pi, mdp, motion, S0=S0, xi=plus, **kw), gamma)
    j_wm_m = return_of(rollout_wm(wm, pi, mdp, motion, S0=S0, xi=minus, **kw), gamma)

    return {
        "dJ_sim": j_sim_p - j_sim_m,
        "dJ_wm": j_wm_p - j_wm_m,
        "obs_only": j_obs_p - j_obs_m,
    }
