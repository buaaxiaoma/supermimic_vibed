# 数据与资产检查（Task 2）

日期：2026-10-30

## 下载

HuggingFace 直连在本机被 TLS 拦截（`curl: (60) self-signed certificate in certificate chain`），
`github.com` 亦不可达。改用镜像 `HF_ENDPOINT=https://hf-mirror.com` 通过 `huggingface_hub.snapshot_download` 完成：

```python
from huggingface_hub import snapshot_download
snapshot_download("lvhaidong/LAFAN1_Retargeting_Dataset", repo_type="dataset",
                  allow_patterns=["g1/*", "robot_description/g1/*"], local_dir="data_raw")
```

## 结果

| 项 | 值 |
|---|---|
| `data/lafan1_g1/*.csv` | 40 |
| `data/g1_description/meshes/*` | 38 |
| URDF | `data/g1_description/g1_29dof_rev_1_0.urdf` |
| MJCF | `data/g1_description/g1_29dof.xml`（nq 36, nv 35, nbody 40, ngeom 71, mass 33.34 kg） |
| USD | `data/g1_usd/g1_29dof/g1_29dof.usda` + `payloads/`（含 `Physics/mujoco.usda`） |

总帧数 264705（2.45 h @ 30 fps）—— 待远程容器内用 `python tools/test_env.py` 复核，
本机因 `isaaclab_newton` API 不匹配无法加载 `MotionLib` 的仿真侧依赖链。

## 未验证

- `tools/test_env.py` 的 FK 一致性（~1e-6）本机未能运行，见 `2026-10-30-smoke-test-4090.md`。
