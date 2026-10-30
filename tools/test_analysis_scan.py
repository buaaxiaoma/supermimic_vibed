"""Sim-side check for ``sensitivity_scan`` — needs the simulator, run on the GPU box.

    python tools/test_analysis_scan.py --num_envs 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--ckpt", required=True)
add_launcher_args(parser)
args = parser.parse_args()

import torch

import supermimic.networks as networks
from supermimic.analysis import sensitivity_scan
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list

dev = "cuda:0"
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))

ck = torch.load(args.ckpt, map_location=dev)
a_ = ck["args"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
pi = Policy(OBS_DIM, 73 * a_["goal_frames"], hist=a_["hist"]).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()
wm = WorldModel().to(dev)
wm.load_state_dict(ck["wm"])
wm.eval()

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        failures.append(name)


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, eval_mode=True, ground_mu=0.1)
    B = args.num_envs
    frames = motion.clip_start[:B].clone()
    xi = torch.zeros(B, 33, device=dev)
    xi[:, 0] = 0.7

    res = sensitivity_scan(env, wm, pi, mdp, motion, frames=frames, xi_base=xi, direction_idx=0,
                           delta=0.05, steps=8, goal_frames=a_["goal_frames"])
    for k in ("dJ_sim", "dJ_wm", "obs_only"):
        check(f"{k}_shape", res[k].shape == (B,), str(res[k].shape))
        check(f"{k}_finite", bool(torch.isfinite(res[k]).all()))

    # friction must be live: with ground_mu=0.1 the effective mu is xi[:,0], so a 0.05 change
    # has to move the simulated return. A zero here means the ground friction is still pinning it.
    check("friction_direction_is_live", bool((res["dJ_sim"].abs() > 0).any()),
          f"|dJ_sim|={res['dJ_sim'].abs().tolist()}")

    # the observation-only arm must exist as its own channel and be distinct from the whole
    check("obs_only_distinct", not torch.allclose(res["obs_only"], res["dJ_sim"]), "obs arm identical to full arm")

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
