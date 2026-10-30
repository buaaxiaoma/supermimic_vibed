"""xi-sensitivity scan: how much does each sim-context direction move the return, in the
simulator versus in the world model?

The go/no-go experiment of the design doc (§8). A direction where the simulator is sensitive and
the world model is not is a direction the policy can exploit during BPTT training.

    python tools/sensitivity.py --ckpt logs/lafan1_main/ckpt_014000.pt --out logs/sensitivity
"""

import argparse
import collections
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", default=os.path.join(REPO, "logs", "sensitivity"))
parser.add_argument("--motion_dir", default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--steps", type=int, default=32, help="control steps per rollout")
parser.add_argument("--delta", type=float, default=0.2,
                    help="perturbation as a fraction of each direction's natural width "
                         "(friction 0.45, joint offset 0.01 rad, CoM 0.05 m — the divisors normalize_xi uses)")
parser.add_argument("--frames_per_class", type=int, default=8)
parser.add_argument("--directions", default="all", help="comma-separated indices, or 'all'")
parser.add_argument("--ground_mu", type=float, default=0.1, help="ground sliding friction; must be < every xi[:,0] tested")
add_launcher_args(parser)
args = parser.parse_args()

import torch

import supermimic.networks as networks
from supermimic.analysis import XI_NAMES, rollout_sim, sensitivity_scan, set_xi
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list

dev = "cuda:0"


def motion_class(name: str) -> str:
    """'fallAndGetUp1_subject1' -> 'fallAndGetUp'."""
    return re.sub(r"\d.*$", "", name)


def xi_width(direction: int) -> float:
    """Natural width of one xi direction — the divisor :func:`normalize_xi` uses."""
    if direction == 0:
        return 0.45  # friction: normalize_xi divides by 0.45
    if 1 <= direction <= 29:
        return 0.01  # joint zero offset, rad
    return 0.05  # torso CoM offset, m


def pick_frames(motion: MotionLib, frames_per_class: int, classes: list[str]) -> dict[str, torch.Tensor]:
    out = {}
    for c in classes:
        ids = [i for i, n in enumerate(motion.names) if motion_class(n) == c]
        if not ids:
            continue
        per = max(1, frames_per_class // len(ids))
        fr = []
        for i in ids:
            s = int(motion.clip_start[i])
            e = int(motion.clip_end[i])
            # stay within the clip so that a `steps`-long rollout does not run past the reference
            e = max(s, e - args.steps - 1)
            fr.append(torch.linspace(s, e, per, device=motion.device).long())
        out[c] = torch.unique(torch.cat(fr))
    return out


kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))

ck = torch.load(args.ckpt, map_location=dev)
a_ = ck["args"]
K, Hh = a_["goal_frames"], a_["hist"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()
wm = WorldModel().to(dev)
wm.load_state_dict(ck["wm"])
wm.eval()

dirs = list(range(33)) if args.directions == "all" else [int(x) for x in args.directions.split(",")]

os.makedirs(args.out, exist_ok=True)
rows = []
classes = sorted({motion_class(n) for n in motion.names})
frames_by_class = pick_frames(motion, args.frames_per_class, classes)
print("classes:", {k: len(v) for k, v in frames_by_class.items()}, flush=True)
max_batch = max((len(v) for v in frames_by_class.values()), default=1)

sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, max_batch, motion, mdp, eval_mode=True, ground_mu=args.ground_mu)
    xi_base = torch.zeros(max_batch, 33, device=dev)
    xi_base[:, 0] = 0.7  # a mid-range friction, well above ground_mu

    for cls, frames in frames_by_class.items():
        n = len(frames)
        pad = max_batch - n
        fr = frames if pad == 0 else torch.cat((frames, frames[-1:].repeat(pad)))
        for d in dirs:
            delta = args.delta * xi_width(d)
            # the ground friction must stay below every effective mu, or MuJoCo's pair-max
            # pins the robot friction and the direction is dead by construction
            assert args.ground_mu < float((xi_base[:, 0] - delta).min()), (
                f"ground_mu={args.ground_mu} would pin the effective friction at "
                f"{args.ground_mu}; it must stay below {float((xi_base[:, 0] - delta).min()):.3f}"
            )
            res = sensitivity_scan(
                env, wm, pi, mdp, motion,
                frames=fr, xi_base=xi_base, direction_idx=d, delta=delta,
                steps=args.steps, goal_frames=K,
            )
            metrics = ["dJ_sim", "dJ_wm", "obs_only"]
            if d == dirs[0]:
                metrics += ["J_base_sim", "J_base_wm"]  # identical for every direction: record once
            for key in metrics:
                vals = res[key][:n]
                for j, v in enumerate(vals.tolist()):
                    rows.append({"class": cls, "direction": d, "name": XI_NAMES[d], "metric": key,
                                 "frame": int(fr[j]), "value": v})
        print(f"[scan] {cls}: {len(frames)} frames x {len(dirs)} directions", flush=True)

json.dump(rows, open(os.path.join(args.out, "sensitivity.json"), "w"))
with open(os.path.join(args.out, "sensitivity.csv"), "w") as f:
    f.write("class,direction,name,metric,frame,value\n")
    for r in rows:
        f.write(f"{r['class']},{r['direction']},{r['name']},{r['metric']},{r['frame']},{r['value']:.6g}\n")

# summary: per-direction mean |dJ_sim| and |dJ_wm|, normalised by the baseline return
agg = collections.defaultdict(lambda: collections.defaultdict(list))
base = collections.defaultdict(list)
for r in rows:
    if r["metric"].startswith("J_base"):
        base[r["metric"]].append(r["value"])
    else:
        agg[r["direction"]][r["metric"]].append(abs(r["value"]))
mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
j_base_sim = mean(base["J_base_sim"])
j_base_wm = mean(base["J_base_wm"])
summary = []
for d in sorted(agg):
    sim, wm_, obs = mean(agg[d]["dJ_sim"]), mean(agg[d]["dJ_wm"]), mean(agg[d]["obs_only"])
    summary.append({
        "direction": d, "name": XI_NAMES[d],
        "abs_dJ_sim": sim, "abs_dJ_wm": wm_, "abs_obs_only": obs,
        "rel_dJ_sim": sim / j_base_sim if j_base_sim else None,
        "rel_dJ_wm": wm_ / j_base_wm if j_base_wm else None,
        "ratio_sim_over_wm": sim / wm_ if wm_ > 0 else None,  # None, not infinity: strict JSON
    })
json.dump({"J_base_sim": j_base_sim, "J_base_wm": j_base_wm, "directions": summary},
          open(os.path.join(args.out, "summary.json"), "w"), indent=1)
for s in summary:
    print(f"{s['direction']:2d} {s['name']:38s} "
          f"|dJ_sim|={s['abs_dJ_sim']:.4g} ({100 * (s['rel_dJ_sim'] or 0):.2f}% J_base) "
          f"|dJ_wm|={s['abs_dJ_wm']:.4g} ({100 * (s['rel_dJ_wm'] or 0):.2f}%)")
print(f"\nJ_base: sim={j_base_sim:.4g} wm={j_base_wm:.4g}")
print(f"wrote {args.out}/sensitivity.{{json,csv}} and summary.json")
