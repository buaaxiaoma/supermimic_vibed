# 远程 8×4090 执行清单（Phase 1）

分支：`feat/phase1-sensitivity`（提交 `ed1cd8f`）

```bash
git fetch origin
git checkout feat/phase1-sensitivity   # 或先在本地 push 后 fetch
```

## 1. 环境（对应 Task 1）

```bash
tools/setup.sh                          # 构建 isaac-lab-kitless + supermimic 镜像
docker images | grep -E 'supermimic|isaac-lab-kitless'
docker/run_container.sh 0 128g
docker exec supermimic python tools/test_env.py
```

`test_env.py` 通过的标准：FK 位置/朝向/速度误差 ~1e-6，退出码 0。
**若失败，先停下**——说明远程镜像与本分支假设不一致，后续结果不可比。

## 2. 数据（对应 Task 2）

```bash
docker exec supermimic bash tools/download_data.sh
docker exec supermimic bash -c 'ls data/lafan1_g1 | wc -l'   # 期望 40
```

本机下载时 HuggingFace 直连被 TLS 拦截，改用 `HF_ENDPOINT=https://hf-mirror.com`。
远程若可直连则无需镜像。

## 3. 基线（对应 Task 3）

```bash
CPUS=0-23 ./run_train.sh lafan1_main
docker exec -it supermimic tmux attach -t lafan1_main
docker exec supermimic python supermimic/evaluate.py \
    --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_014000.pt
```

记录 `norm_return`。**判定改成健康门槛**（不再对标 README 的 0.829，因为版本不同）：

- `wm_loss` 单调下降、`imag_rew` 上升、`surv` 不为 0；
- eval `norm_return` 明显高于随机水平，`term_frac` 不接近 1；
- 无 NaN。

本机训出的这个数字就是本文所有对比的分母。论文中须写明 Isaac Lab / newton 的 commit，
且**不得把 sim 数字与 SuperMimic 论文的 0.82 直接比较**。

## 4. 仿真侧测试（对应 Task 4、6）

这两项在本机**未能运行**（`isaaclab_newton` 与源码树 API 不一致），必须在固定版本镜像里跑：

```bash
docker exec supermimic python tools/test_analysis_sim.py --num_envs 4
docker exec supermimic python tools/test_analysis_scan.py --num_envs 2 \
    --ckpt logs/lafan1_main/ckpt_014000.pt
```

`test_analysis_sim.py` 必须全部 PASS，其中 `ground_mu_below_xi` 与
`robot_mu_written` 两项**决定摩擦方向是否真的被扰动**——若这两项失败，
后面的敏感性结论全部作废。

## 5. go/no-go 扫描（对应 Task 6）

```bash
docker exec supermimic python tools/sensitivity.py \
    --ckpt logs/lafan1_main/ckpt_014000.pt \
    --out logs/sensitivity \
    --steps 32 --delta 0.2 --frames_per_class 8
```

`--delta` 是**每个方向自然宽度的比例**（摩擦 0.45、关节零位 0.01 rad、躯干 CoM 0.05 m —— 即
`normalize_xi` 用的除数），所以 0.2 表示所有方向在 WM 输入空间里移动同样的量。不要给各方向
设不同的绝对扰动，那会让摩擦方向看起来"不敏感"。

产出 `logs/sensitivity/{sensitivity.json,sensitivity.csv,summary.json}`；
`summary.json` 含 `J_base_sim` / `J_base_wm` 与每个方向的 `rel_dJ_sim` / `rel_dJ_wm`。

判定（写入 `docs/superpowers/notes/2026-10-30-go-no-go.md`）：

- 至少 2 个方向满足 `rel_dJ_sim > 0.05` 且 `rel_dJ_wm < 0.01` → **GO**
- 否则 → **NO-GO**，按 spec §6 第三行换靶心

## 6. 已知的本地/远程差异

| 项 | 本机 | 远程 |
|---|---|---|
| GPU | RTX 5060 8 GB | 8× 4090 24 GB |
| Isaac Lab | 用户全局安装（版本不匹配） | 仓库 pin 的 3.0.0 @ 46d6cc3 |
| 训练 | 不可能 | 基线 + 后续实验 |
| WM 侧单测 | 通过（15/15） | 应同样通过 |
