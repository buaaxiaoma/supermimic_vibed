"""Sim-side checks for ``supermimic.analysis``: xi layout, effective friction/CoM, deterministic rollout.

Run inside the Isaac Lab 3 venv (or the container):

    python tools/test_analysis_sim.py --num_envs 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
add_launcher_args(parser)
args = parser.parse_args()

import torch
import warp as wp

from supermimic.analysis import XI_NAMES, Rollout, rollout_sim, set_xi
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import default_joint_pos_list

dev = "cuda:0"
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- layout
check("xi_names_length", len(XI_NAMES) == 33, f"got {len(XI_NAMES)}")
check("xi_names_friction", XI_NAMES[0] == "friction", XI_NAMES[0])
check("xi_names_joint", XI_NAMES[1] == "joint_offset:left_hip_pitch_joint", XI_NAMES[1])
check("xi_names_com", XI_NAMES[30:33] == ["com:x", "com:y", "com:z"], str(XI_NAMES[30:33]))

sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    # ground mu pinned low so that xi[0] is the effective pair maximum
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, eval_mode=True, ground_mu=0.1)

    # ------------------------------------------------------------ friction
    xi = env.xi.clone()
    xi[:, 0] = 0.45
    set_xi(env, xi)
    mu = torch.as_tensor(env._mu_view)
    check("robot_mu_written", torch.allclose(mu, torch.full_like(mu, 0.45), atol=1e-6), f"max dev {(mu - 0.45).abs().max():.2e}")

    model = env.sim.backend_manager.get_model()
    mu_all = wp.to_torch(model.shape_material_mu)
    shape_body = wp.to_torch(model.shape_body)
    ground = mu_all[shape_body < 0]
    check("ground_mu_below_xi", bool((ground <= 0.45).all()), f"ground max {float(ground.max()):.3f}")

    # ------------------------------------------------------------ torso com
    xi2 = env.xi.clone()
    xi2[:, 30:33] = torch.tensor([0.02, -0.02, 0.03], device=dev)
    set_xi(env, xi2)
    com = env.robot.data.body_com_pose_b.torch[:, env.torso_id, :3]
    expected = env.default_torso_com + xi2[:, 30:33]
    check("torso_com_written", torch.allclose(com, expected, atol=1e-5), f"max dev {(com - expected).abs().max():.2e}")

    # ------------------------------------------------------------ rollout
    frames = motion.clip_start[: args.num_envs].clone()
    xi3 = env.xi.clone()
    xi3[:, 0] = 0.6
    xi3[:, 1:30] = 0.0
    xi3[:, 30:33] = 0.0
    out = rollout_sim(env, pi=None, mdp=mdp, motion=motion, frames=frames, xi=xi3, steps=8, action_fn=lambda _obs, _f: torch.zeros(env.N, 29, device=dev))
    check("rollout_is_rollout", isinstance(out, Rollout))
    check("rollout_state_shape", out.S[0].shape == (9, args.num_envs, 3), str(out.S[0].shape))
    check("rollout_action_shape", out.a.shape == (8, args.num_envs, 29), str(out.a.shape))
    check("rollout_reward_finite", bool(torch.isfinite(out.r).all()))
    check("xi_unchanged", torch.allclose(env.xi[:, 0], torch.full((env.N,), 0.6, device=dev)), f"got {env.xi[0, 0]:.3f}")

    # ------------------------------------------------------------ determinism
    out2 = rollout_sim(env, pi=None, mdp=mdp, motion=motion, frames=frames, xi=xi3, steps=8, action_fn=lambda _obs, _f: torch.zeros(env.N, 29, device=dev))
    check("rollout_deterministic", torch.allclose(out.S[0], out2.S[0], atol=1e-6))

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
