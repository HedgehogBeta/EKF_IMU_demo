#!/usr/bin/env python3
"""题面 §4(4) 要求的曲线：Roll/Pitch/Yaw–Time，以及选做位置估计的 x/y/z–Time。

用法（不依赖 ROS）：
    python3 scripts/plot.py output/ekf_pv_out.csv data/pose_cov.csv [输出目录]

出图：
    01_rpy_time.png       Roll/Pitch/Yaw–Time 三条曲线（估计 vs FAST-LIO 观测）
    02_position_time.png  x/y/z–Time 三条曲线（估计 vs 观测）

本脚本只负责出图；数值验收（姿态测地角误差、位置跟踪、NIS、b_a 等）见
scripts/eval_eskf_pv.py。
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# matplotlib 认不出 Noto CJK 的 .ttc 集合，需手动注册字体文件
_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_NOTO).exists():
    fm.fontManager.addfont(_NOTO)  # 只注册第一张脸（JP），字形覆盖含简体
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def read_csv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        rows = np.array(
            [[float(v) for v in line.strip().split(",")] for line in f if line.strip()]
        )
    return rows, idx


def slerp(q0, q1, u):
    """四元数球面插值（xyzw）。q0/q1 为 (...,4)，u 标量或 (...,1)。"""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)  # 符号对齐：q 与 -q 是同一旋转，不展开会走长弧
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-8
    w1 = np.where(near, u, np.sin((1 - u) * theta) / np.where(near, 1, sin_theta))
    w2 = np.where(near, 1 - u, np.sin(u * theta) / np.where(near, 1, sin_theta))
    q = w1 * q0 + w2 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_xyzw_to_rpy_deg(q):
    """xyzw -> ZYX 欧拉角（度）"""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(np.stack([roll, pitch, yaw], axis=-1))


def align_pose_to(t_q, q_p, t_p):
    """把 pose 观测 SLERP 到给定时刻：找相邻两帧，前不足处保持第一帧。"""
    j = np.clip(np.searchsorted(t_p, t_q) - 1, 0, len(t_p) - 2)
    span = t_p[j + 1] - t_p[j]
    u = np.where(span > 0, (t_q - t_p[j]) / np.where(span > 0, span, 1.0), 0.0)
    u = np.clip(u, 0.0, 1.0)
    return slerp(q_p[j], q_p[j + 1], u[:, None])


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ekf_path, pose_path = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(ekf_path))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    e, ei = read_csv(ekf_path)
    p, pi = read_csv(pose_path)

    t = e[:, ei["time"]]
    t0 = t[0]
    tr = t - t0
    q_e = np.stack([e[:, ei[c]] for c in ("qw", "qx", "qy", "qz")], axis=1)[:, [1, 2, 3, 0]]
    t_p = p[:, pi["time"]]
    q_p = np.stack([p[:, pi[c]] for c in ("qx", "qy", "qz", "qw")], axis=1)

    q_ref = align_pose_to(t, q_p, t_p)
    rpy_e = quat_xyzw_to_rpy_deg(q_e)
    rpy_ref = quat_xyzw_to_rpy_deg(q_ref)

    missing = [c for c in ("x", "y", "z") if c not in ei]
    if missing:
        sys.exit(f"{ekf_path}: 缺少列 {missing}（位置估计未输出）")
    p_hat = np.stack([e[:, ei[c]] for c in ("x", "y", "z")], axis=1)
    p_obs = np.stack([np.interp(t, t_p, p[:, pi[c]]) for c in ("x", "y", "z")], axis=1)

    dt = float(np.median(np.diff(t)))
    print(f"帧数 {len(t)}（dt 中位 {dt * 1e3:.3f} ms），观测 {len(t_p)} 帧")
    print(f"时长 {tr[-1]:.1f} s；轨迹合模长最大 {np.linalg.norm(p_hat, axis=1).max():.2f} m")

    # ---------------- 01：Roll/Pitch/Yaw – Time（题面要求）----------------
    names = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(tr, rpy_e[:, k], lw=0.9, color="C0", label="ESKF 估计")
        ax.plot(tr, rpy_ref[:, k], lw=0.7, color="C2", ls="--", alpha=0.8,
                label="FAST-LIO 观测")
        ax.set_ylabel(f"{names[k]} (°)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("Roll / Pitch / Yaw – Time（题面 §4(4) 要求的三条曲线）")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_rpy_time.png"), dpi=150)
    plt.close(fig)

    # ---------------- 02：x/y/z – Time（选做位置估计）----------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(tr, p_hat[:, k], lw=0.9, color="C0", label="ESKF 估计（15 维）")
        ax.plot(tr, p_obs[:, k], lw=0.8, color="C1", ls="--", alpha=0.8,
                label="FAST-LIO 位置观测")
        ax.set_ylabel(f"{'xyz'[k]} (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("选做：位置 x/y/z – Time（机器人 60–100 s 绕圈，最后回到原点）")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "02_position_time.png"), dpi=150)
    plt.close(fig)

    print(f"图已存: {fig_dir}/01_rpy_time.png, 02_position_time.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
