"""World-model-side checks for ``supermimic.analysis`` — pure torch, no simulator needed.

    python tools/test_analysis_wm.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from supermimic.analysis import Rollout, return_of, rollout_wm
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.robot import default_joint_pos_list

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        failures.append(name)


wm = WorldModel(hidden=(64, 64)).to(dev).eval()
B = 2
frames = motion.clip_start[:B].clone()
S0 = tuple(x.clone() for x in motion.ref_state(frames))
xi = torch.zeros(B, 33, device=dev)
xi[:, 0] = 0.6
xi[:, 1:30] = 0.002
zero_action = lambda _hist, _f: torch.zeros(B, 29, device=dev)  # noqa: E731

# ---------------------------------------------------------------- one step matches the model
out = rollout_wm(wm, None, mdp, motion, S0=S0, xi=xi, frames=frames, steps=1, action_fn=zero_action)
S1 = wm.step(S0, torch.zeros(B, 29, device=dev), normalize_xi(xi))
for name, got, want in zip(("p", "R", "v", "w", "q", "qd"), out.S, S1):
    err = (got[1] - want).abs().max().item()
    check(f"one_step_{name}", err < 1e-6, f"max err {err:.2e}")

# ---------------------------------------------------------------- shapes and bookkeeping
out = rollout_wm(wm, None, mdp, motion, S0=S0, xi=xi, frames=frames, steps=4, action_fn=zero_action)
check("rollout_is_rollout", isinstance(out, Rollout))
check("state_shape", out.S[0].shape == (5, B, 3), str(out.S[0].shape))
check("action_shape", out.a.shape == (4, B, 29), str(out.a.shape))
check("reward_shape", out.r.shape == (4, B), str(out.r.shape))
check("term_is_bool", out.term.dtype == torch.bool, str(out.term.dtype))
check("reward_finite", bool(torch.isfinite(out.r).all()))

# ---------------------------------------------------------------- termination masking
# rewards after the first termination must be zero; alive must be false from then on
dead = ~out.term.cumsum(0).bool().squeeze(-1) if out.term.any() else None
if out.term.any():
    first = out.term.float().argmax(0)
    for b in range(B):
        t0 = int(first[b])
        if t0 + 1 < out.r.shape[0]:
            check(f"reward_masked_after_term_{b}", bool((out.r[t0 + 1 :, b] == 0).all()))

# ---------------------------------------------------------------- determinism
out2 = rollout_wm(wm, None, mdp, motion, S0=S0, xi=xi, frames=frames, steps=4, action_fn=zero_action)
check("deterministic", torch.allclose(out.S[0], out2.S[0], atol=1e-6))

# ---------------------------------------------------------------- return arithmetic
r = torch.ones(3, 2)
alive = torch.ones(3, 2)
check("return_gamma1", torch.allclose(return_of(Rollout(S=out.S, a=out.a, r=r, alive=alive, term=out.term), 1.0),
                                      torch.full((2,), 3.0)), str(return_of(Rollout(S=out.S, a=out.a, r=r, alive=alive, term=out.term), 1.0)))
check("return_gamma0", torch.allclose(return_of(Rollout(S=out.S, a=out.a, r=r, alive=alive, term=out.term), 0.0),
                                      torch.full((2,), 1.0)))

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
