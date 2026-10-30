# Phase 1: 基础设施 + 敏感性赌注实验 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在远程 8×4090 上跑通基线，并完成设计文档 §8 的 go/no-go 实验——测量 ξ 扰动在物理仿真与可微世界模型中的收益敏感度差异。

**Architecture:** 新增 `supermimic/analysis.py` 提供确定性的策略 rollout（仿真侧与 WM 侧）与返回计算；`tools/sensitivity.py` 遍历 ξ 方向 × 动作类别 × 留出 checkpoint，输出 JSON + 图。仿真侧复用 `TrackingSim`，WM 侧复用 `WorldModel.step` 与 `MDP.reward`。

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab 3 kit-less + Newton/MuJoCo-Warp, pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-10-28-model-error-budget-design.md`

> **为什么这是 Phase 1：** 设计文档 §8 明确该实验决定整篇论文的真假。若敏感度差异不显著，主线需换靶心（spec §6 第三行）。因此先交付"能跑 + 能判死"的最小闭环；G0 的完整 ξ 扩展、G1 误差预算、G2 可辨识性、G3/G4 各自另起计划，待本阶段结论后再写。

## 执行环境

代码在本机（`/home/mjy/projects/supermimic_vibed`，origin `https://github.com/zita-ch/supermimic_vibed.git`）编写并提交；**所有需要 GPU 仿真器或训练的步骤在远程 8×4090 上执行**。原因（实测）：

- 本机 GPU 为 RTX 5060 Laptop，**8 GB 显存**；基线训练（4096 envs + 1000 步回放缓冲 + 32 步 BPTT）实测估计需 10–20 GB。
- 本机 `/` 仅剩 **17 GB** 空闲，而 kit-less 镜像 + Isaac Lab 需 ~20 GB，本地构建大概率失败。

因此：

- Task 5（WM rollout）的测试是纯 torch + 数据，**可在本机跑**（需先 `tools/download_data.sh`，约 0.2 GB）。
- Task 1–4、6 的测试与运行**必须在远程容器内**。
- 交付方式：本机 `git commit && git push`，远程 `git pull` 后执行。每次提交前确认 `docs/` 与代码一并推送。

## Global Constraints

- 机器人固定为 Unitree G1 29-DoF（`g1_29dof_rev_1_0`），关节顺序以 `supermimic.robot.G1_JOINT_NAMES` 为准。
- 控制频率 50 Hz（`supermimic.motion.CONTROL_DT`），仿真 4 子步。
- ξ 布局不可改动：`[接触摩擦 μ (1), 关节零位 (29), 躯干 CoM (3)]`，`XI_DIM == 33`。
- 策略与 critic 只在训练中产生；本阶段只加载 checkpoint，不训练新策略。
- 所有会改动基线的开关必须有默认值保持现有行为，且默认路径下 `python supermimic/evaluate.py --ckpt <ckpt>` 的数字不变。
- 评估判据与 spec 一致：终止条件为 `mdp.termination`；失败动作数与原 `evaluate.py` 可比。
- 测试不得依赖网络；需要 GPU/仿真器的测试必须在容器内运行。
- 任何对比实验必须用**同一** `mdp.reward`、同一 FK，且**无 critic bootstrap**。

## Review Focus

按最可能咬人的顺序：

1. **ξ[0] 在评估模式下完全失效。** `TrackingSim._setup_friction` 在 `eval_mode=True` 时把地面 μ 设为 1.0，MuJoCo 取配对 `max`，因此 ξ[0] ≤ 1.0 的任何扰动都不改变物理。一个"摩擦方向无敏感度"的结论可能纯是伪影。→ Task 4 的测试必须证明扰动确实改变了有效 μ。
2. **评估模式不重采关节零位。** `reset()` 仅在 `not eval_mode` 时重采 `xi[:,1:30]`，所以零位在 rollout 期间固定；若测试以为它会变，结果无法解释。→ Task 4 的测试固定并断言 ξ 在 rollout 内不变。
3. **WM 侧与仿真侧奖励来源不一致会导致 ΔJ 混淆。** 两侧必须都用 `mdp.reward` 的解析奖励。→ Task 5 的测试断言同一步的奖励一致到数值精度。
4. **ξ 同时进入动力学与观测。** 关节零位既改变动作目标（`default_q + xi[:,1:30] + alpha*a`）又改变观测（`q - default_q - xi[:,1:30]`）。敏感性因此包含"策略看到的世界变了"和"物理变了"两部分。→ Task 6 必须在报告里分开，测试用"只改观测不改物理"的对照臂验证分离可行。
5. **仿真用硬终止，WM 训练用软终止。** 比较收益时必须统一用硬终止；否则长视界下 WM 的收益被系统性抬高。→ Task 5 的测试断言 rollout 使用 `mdp.termination` 而非 `soft_termination`。

---

### Task 1: 在远程 4090 上验证仿真栈

**Files:**
- Create: `docs/superpowers/notes/2026-10-30-smoke-test-4090.md`

**Interfaces:**
- Consumes: 现有 `docker/`、`tools/setup.sh`、`tools/test_env.py`
- Produces: 一份结论记录，说明 Isaac Lab 3 kit-less + MuJoCo-Warp 在 Ampere 上是否可跑，以及实测吞吐

- [ ] **Step 1: 确认远程机器的驱动与容器**

Run:
```bash
ssh <remote> 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'
docker --version && docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
```
Expected: 列出 8 × RTX 4090，24 GB；驱动支持 CUDA 12.x 以上。

- [ ] **Step 2: 构建镜像**

Run:
```bash
tools/setup.sh
docker images | grep supermimic
```
Expected: `isaac-lab-kitless:main-46d6cc3` 与 `supermimic:latest` 均存在。

- [ ] **Step 3: 启动容器并做一次物理有效性检查**

Run:
```bash
docker/run_container.sh 0 128g
docker exec supermimic python tools/test_env.py
```
Expected: FK 与仿真器一致到 ~1e-6，脚本以退出码 0 结束。

- [ ] **Step 4: 实测吞吐**

Run:
```bash
docker exec supermimic python tools/bench.py --num_envs 1024
docker exec supermimic python tools/bench.py --num_envs 2048
```
Expected: 打印每档 iterations/s 与显存占用。记录 1024 与 2048 的数字。

- [ ] **Step 5: 记录结论**

把以下写进 `docs/superpowers/notes/2026-10-30-smoke-test-4090.md`：驱动/CUDA 版本；`test_env.py` 是否通过；两档 `num_envs` 的 it/s 与显存；按 1024 envs 估算到达 14000 迭代所需小时数；若失败，记录确切报错与已排查项。

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/notes/2026-10-30-smoke-test-4090.md
git commit -m "docs: record 4090 smoke-test result"
```

---

### Task 2: 准备数据与机器人资产

**Files:**
- Modify: 无（使用 `tools/download_data.sh`）
- Create: `docs/superpowers/notes/2026-10-30-data-check.md`

**Interfaces:**
- Consumes: Task 1 的容器
- Produces: `data/lafan1_g1/*.csv`（40 个）、`data/g1_description/g1_29dof.xml`、`data/g1_usd/g1_29dof/g1_29dof.usda`

- [ ] **Step 1: 下载并转换资产**

Run:
```bash
docker exec supermimic bash tools/download_data.sh
ls data/lafan1_g1 | wc -l
```
Expected: `40`。

- [ ] **Step 2: 校验总帧数**

Run:
```bash
docker exec supermimic python - <<'PY'
import glob, csv
n = sum(sum(1 for _ in open(f)) - 1 for f in glob.glob('data/lafan1_g1/*.csv'))
print(n)
PY
```
Expected: `264705`（2.45 h @ 30 fps）。

- [ ] **Step 3: 记录资产状态**

写 `docs/superpowers/notes/2026-10-30-data-check.md`：40 个 clip、总帧数、USD 路径、`tools/test_env.py` 的 FK 一致性数字。

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/notes/2026-10-30-data-check.md
git commit -m "docs: record LAFAN1 data and asset check"
```

---

### Task 3: 复现基线 checkpoint

**Files:**
- Create: `logs/lafan1_main/`（不入库）、`docs/superpowers/notes/2026-10-30-baseline.md`

**Interfaces:**
- Consumes: Task 2 的数据
- Produces: `logs/lafan1_main/ckpt_014000.pt` 等 checkpoint，供 Task 6 使用；一条基线 `norm_return`

- [ ] **Step 1: 启动训练**

Run:
```bash
CPUS=0-23 ./run_train.sh lafan1_main
docker exec -it supermimic tmux attach -t lafan1_main
```
Expected: 日志出现 `it 50 | ... it/s`，`wm` 损失单调下降。

- [ ] **Step 2: 训练到至少 14000 迭代**

Run:
```bash
docker exec supermimic tail -n 3 logs/lafan1_main/train_log.csv
```
Expected: 最后一行的 iteration ≥ 14000。

- [ ] **Step 3: 评估基线**

Run:
```bash
docker exec supermimic python supermimic/evaluate.py --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_014000.pt
```
Expected: 打印 `norm_return`、`n_failed`、`failed`。**记录该数字**；后续所有对比都以它为参照。

- [ ] **Step 4: 记录并判定**

写 `docs/superpowers/notes/2026-10-30-baseline.md`：GPU 型号、`num_envs`、到达 14000 迭代的墙钟小时、`norm_return`、失败动作列表。若 `norm_return` 与本仓库 README 的 0.829 相差超过 0.05，停止并报告——这说明环境有偏差，后续敏感性结论不可比。

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/2026-10-30-baseline.md
git commit -m "docs: record baseline reproduction"
```

---

### Task 4: 可显式设定 ξ 的确定性仿真 rollout

**Files:**
- Create: `supermimic/analysis.py`
- Modify: `supermimic/env.py`（`TrackingSim.__init__` 增加 `ground_mu` 参数，默认值保持现有行为）
- Test: `tools/test_analysis_sim.py`

**Interfaces:**
- Consumes: `TrackingSim`, `MDP`, `MotionLib`, `Policy`
- Produces:
  - `XI_NAMES: list[str]`，长度 33；`XI_NAMES[0] == "friction"`，`XI_NAMES[1:30]` 为 `f"joint_offset:{name}"`，`XI_NAMES[30:33]` 为 `f"com:{ax}"`（`ax ∈ {"x","y","z"}`）
  - `set_xi(env: TrackingSim, xi: torch.Tensor) -> None`：写入 ξ、更新机器人形状 μ、推进一次 model change
  - `rollout_sim(env, pi, mdp, motion, *, frames: torch.Tensor, xi: torch.Tensor, steps: int) -> Rollout`
  - `@dataclass Rollout`：`S: tuple[torch.Tensor, ...]`（各 `(steps+1, B, ...)`），`a: (steps, B, 29)`，`r: (steps, B)`，`alive: (steps, B)`，`term: (steps, B)`

- [ ] **Step 1: 写失败测试**

```python
# tools/test_analysis_sim.py
import torch
from supermimic.analysis import XI_NAMES, set_xi, rollout_sim, Rollout

def test_xi_names_layout():
    assert len(XI_NAMES) == 33
    assert XI_NAMES[0] == "friction"
    assert XI_NAMES[1] == "joint_offset:left_hip_pitch_joint"
    assert XI_NAMES[30:33] == ["com:x", "com:y", "com:z"]

def test_set_xi_changes_effective_friction(env):
    import warp as wp
    xi = env.xi.clone()
    xi[:, 0] = 0.45
    set_xi(env, xi)
    # 两个前提都必须成立，否则扰动不改变物理：
    # (1) 机器人形状 mu 被写入 0.45；(2) 地面 mu <= 0.45（MuJoCo 取配对 max）
    assert torch.allclose(torch.as_tensor(env._mu_view),
                          torch.full_like(torch.as_tensor(env._mu_view), 0.45), atol=1e-6)
    model = env.sim.backend_manager.get_model()
    mu_all = wp.to_torch(model.shape_material_mu)
    shape_body = wp.to_torch(model.shape_body)
    ground = mu_all[shape_body < 0]
    assert bool((ground <= 0.45).all()), f"ground mu {ground} pins effective friction"

def test_rollout_preserves_xi(env, pi, mdp, motion):
    frames = motion.clip_start[:4].clone()
    xi = env.xi.clone(); xi[:, 0] = 0.6
    out = rollout_sim(env, pi, mdp, motion, frames=frames, xi=xi, steps=16)
    assert isinstance(out, Rollout)
    assert out.S[0].shape == (17, 4, 3)
    assert torch.allclose(env.xi[:, 0], torch.full((env.N,), 0.6, device=env.device))
```

`env` fixture 在容器内构造 `TrackingSim(sim_cfg, 4, motion, mdp, eval_mode=True, ground_mu=0.1)` 并调用 `env.refresh()`。

- [ ] **Step 2: 运行并确认失败**

Run: `docker exec supermimic python -m pytest tools/test_analysis_sim.py -x -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'supermimic.analysis'`。

- [ ] **Step 3: 实现 `ground_mu` 参数**

在 `TrackingSim.__init__` 签名末尾加 `ground_mu: float | None = None`；**在调用 `self._setup_friction()` 之前**赋值 `self.ground_mu = ground_mu`；`_setup_friction` 中的 `0.3 if not self.eval_mode else 1.0` 改为 `self.ground_mu if self.ground_mu is not None else (0.3 if not self.eval_mode else 1.0)`。

- [ ] **Step 4: 实现 `supermimic/analysis.py` 的仿真部分**

`XI_NAMES` 由 `G1_JOINT_NAMES` 生成。`set_xi` 写 `env.xi[:] = xi`、调 `env._apply_friction()`。`rollout_sim` 逐帧 `env.step(pi(hist, goal))`，用 `mdp.termination` 标记 `term`，用 `~alive_prev` 屏蔽已终止样本，返回 `Rollout`；观测历史与 `evaluate.py` 的初始化方式一致（首帧复制 `Hh` 次）。

- [ ] **Step 5: 运行并确认通过**

Run: `docker exec supermimic python -m pytest tools/test_analysis_sim.py -q`
Expected: 3 passed。

- [ ] **Step 6: 确认默认行为未变**

Run: `docker exec supermimic python supermimic/evaluate.py --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_014000.pt`
Expected: `norm_return` 与 Task 3 Step 3 记录的完全一致。

- [ ] **Step 7: Commit**

```bash
git add supermimic/analysis.py supermimic/env.py tools/test_analysis_sim.py
git commit -m "feat: deterministic sim rollout with explicit xi"
```

---

### Task 5: WM 侧 rollout

**Files:**
- Modify: `supermimic/analysis.py`
- Test: `tools/test_analysis_wm.py`

**Interfaces:**
- Consumes: `WorldModel.step`, `MDP.reward`, `MDP.termination`, `normalize_xi`, `unflatten_state`, `flatten_state`
- Produces: `rollout_wm(wm, pi, mdp, motion, *, S0, xi, steps) -> Rollout`

- [ ] **Step 1: 写失败测试**

```python
# tools/test_analysis_wm.py
import torch
from supermimic.analysis import rollout_wm, Rollout
from supermimic.networks import WorldModel
from supermimic.mdp import normalize_xi

def test_one_step_matches_world_model(mdp, motion, make_state):
    wm = WorldModel().eval()
    S0 = make_state(batch=2)
    a = torch.zeros(2, 29)
    xi = torch.zeros(2, 33); xi[:, 0] = 0.5; xi[:, 1:30] = 0.002
    out = rollout_wm(wm, policy=lambda h, g: a, mdp=mdp, motion=motion,
                     S0=S0, xi=xi, steps=1)
    S1 = wm.step(S0, a, normalize_xi(xi))
    for got, want in zip(out.S, S1):
        assert torch.allclose(got[1], want, atol=1e-6)

def test_wm_uses_hard_termination(mdp, motion, make_state):
    wm = WorldModel().eval()
    S0 = make_state(batch=1)
    out = rollout_wm(wm, policy=lambda h, g: torch.zeros(1, 29), mdp=mdp,
                     motion=motion, S0=S0, xi=torch.zeros(1, 33), steps=4)
    # term 必须是布尔，且来自 mdp.termination 的阈值
    assert out.term.dtype == torch.bool
```

`make_state` fixture 从 `MotionLib.ref_state(frame)` 构造一批状态。

- [ ] **Step 2: 运行并确认失败**

Run: `docker exec supermimic python -m pytest tools/test_analysis_wm.py -x -q`
Expected: FAIL，`ImportError: cannot import name 'rollout_wm'`。

- [ ] **Step 3: 实现 `rollout_wm`**

循环 `steps`：`a = pi(obs_hist, motion.goal(f, K))`；`S1 = wm.step(S, a, normalize_xi(xi))`；`r = mdp.reward(S1, f+1, a, a_prev)`；`term = mdp.termination(S1, f+1)`；`obs` 用 `mdp.obs(S1, a, xi, f+1)` 且**不加噪声**（评估语义）。不做 critic bootstrap。

- [ ] **Step 4: 运行并确认通过**

Run: `docker exec supermimic python -m pytest tools/test_analysis_wm.py -q`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add supermimic/analysis.py tools/test_analysis_wm.py
git commit -m "feat: world-model rollout sharing the analytic reward"
```

---

### Task 6: 敏感性扫描与 go/no-go 报告

**Files:**
- Modify: `supermimic/analysis.py`（加 `sensitivity_scan`）
- Create: `tools/sensitivity.py`
- Test: `tools/test_analysis_scan.py`

**Interfaces:**
- Consumes: Task 4、Task 5 的 `rollout_sim` / `rollout_wm`
- Produces:
  - `sensitivity_scan(env, wm, pi, mdp, motion, *, frames, xi_base, direction_idx, delta, steps) -> dict`，键含 `"dJ_sim"`、`"dJ_wm"`、`"obs_only"`（各为 `(B,)`）
  - CLI `tools/sensitivity.py --ckpt <path> --out <dir> --steps 32 --delta 0.1 --frames-per-class 8`

- [ ] **Step 1: 写失败测试**

```python
# tools/test_analysis_scan.py
import torch
from supermimic.analysis import sensitivity_scan

def test_scan_reports_both_sides_and_obs_only(env, wm, pi, mdp, motion):
    frames = motion.clip_start[:2].clone()
    xi = env.xi.clone(); xi[:, 0] = 0.7
    out = sensitivity_scan(env, wm, pi, mdp, motion, frames=frames, xi_base=xi,
                           direction_idx=0, delta=0.05, steps=8)
    for k in ("dJ_sim", "dJ_wm", "obs_only"):
        assert out[k].shape == (2,)
        assert torch.isfinite(out[k]).all()
```

`obs_only` 是"只改观测不改物理"的对照臂：仿真物理保持 `xi_base`，但把 `mdp.obs` 传入的 ξ 换成扰动值。

- [ ] **Step 2: 运行并确认失败**

Run: `docker exec supermimic python -m pytest tools/test_analysis_scan.py -x -q`
Expected: FAIL，`ImportError`。

- [ ] **Step 3: 实现 `sensitivity_scan`**

对 `xi_base ± delta` 各跑一次 `rollout_sim` 与 `rollout_wm`，计算 `J = Σ γ^t r_t · alive_t`，取差值。`obs_only` 臂：物理 ξ 固定为 `xi_base`，仅在策略观测与奖励参考中使用扰动后的 ξ。注意 **ξ[0] 的有效性**：调用前用 `set_xi(env, ...)` 且 `ground_mu < min(xi[:,0])`，否则 ΔJ 恒为 0。

- [ ] **Step 4: 运行并确认通过**

Run: `docker exec supermimic python -m pytest tools/test_analysis_scan.py -q`
Expected: 1 passed。

- [ ] **Step 5: 实现 CLI**

`tools/sensitivity.py`：加载 checkpoint → 构造 `TrackingSim(..., eval_mode=True, ground_mu=0.1)` → 对 33 个方向 × 5 个动作类别（walk / run / jump / fallAndGetUp / fight，按 `motion.names` 前缀分组）× `--frames-per-class` 帧，跑 `sensitivity_scan`，写 `sensitivity.json` 与 `sensitivity.csv`，并画 `dJ_wm` vs `dJ_sim` 散点。

- [ ] **Step 6: 在远程机上跑完整扫描**

Run:
```bash
docker exec supermimic python tools/sensitivity.py --ckpt logs/lafan1_main/ckpt_014000.pt --out logs/sensitivity --steps 32 --delta 0.1 --frames-per-class 8
```
Expected: 生成 33 × 5 × 8 行结果；wall time 记录在输出里。

- [ ] **Step 7: 写 go/no-go 报告**

写 `docs/superpowers/notes/2026-10-30-go-no-go.md`，必须包含：

1. `dJ_wm` 与 `dJ_sim` 的 Spearman 相关（全方向）与逐方向散点；
2. 哪些方向的 `|dJ_sim|` 显著而 `|dJ_wm|` 接近 0（即疑似被利用）；
3. `obs_only` 臂的贡献占比——若 `obs_only` 已解释大部分 `dJ_sim`，则敏感性主要来自观测通道而非物理，结论需改写；
4. **判定**：若存在 ≥ 2 个方向满足 `|dJ_sim| > 5%·J_base` 且 `|dJ_wm| < 1%·J_base`，判定 GO；否则 NO-GO，并按 spec §6 第三行提出换靶心方案。

- [ ] **Step 8: Commit**

```bash
git add supermimic/analysis.py tools/sensitivity.py tools/test_analysis_scan.py docs/superpowers/notes/2026-10-30-go-no-go.md
git commit -m "feat: xi sensitivity scan and go/no-go report"
```

---

## Self-Review

**Spec coverage（本阶段范围内）**：spec §8 的最小可证伪实验→ Task 6；§6 的 4090 风险→ Task 1；§4.0 的摩擦死区→ Task 4 与 Review Focus 1；§4.3 的"off-distribution"要求→ Task 6 使用参考动作帧；§2 反目标"结论依赖单一动作"→ Task 6 的 5 个动作类别。G0 完整 ξ 扩展、G1/G2/G3/G4 不在本计划内，属后续独立计划。

**未覆盖并有意推迟**：真机采集与动捕（G4）、ξ 扩展（G0）、误差预算（G1）、可辨识性（G2）。理由见头部说明。

**Type consistency**：`Rollout` 的字段在 Task 4 定义、Task 5/6 复用；`XI_NAMES` 在 Task 4 定义、Task 6 的 CLI 使用；`set_xi` 在 Task 4 定义并在 Task 6 Step 3 调用。
