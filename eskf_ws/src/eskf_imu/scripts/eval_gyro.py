#!/usr/bin/env python3
"""M2 验收：纯陀螺积分 vs FAST-LIO 姿态的测地角误差。

把 pose_cov.csv 的四元数（qx,qy,qz,qw，注意标量在最后）按 timestamp
SLERP 插值到 gyro_out.csv 的各帧时刻（q 与 -q 同一旋转，SLERP 前先符号对齐；
pose_cov 比 imu 晚 0.170 s 起步，之前的时刻保持第一帧），然后逐帧算
测地角误差 angle = 2*arccos(|q1·q2|)，报统计量并出图。

用法: eval_gyro.py <gyro_out.csv> <pose_cov.csv> [输出目录]
不依赖 ROS。
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# matplotlib 认不出 Noto CJK 的 .ttc 集合，需手动注册字体文件（与 data_check.py 同法）
_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_NOTO).exists():
    fm.fontManager.addfont(_NOTO)  # 只注册第一张脸（JP），字形覆盖含简体
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

THRESHOLD_DEG = 5.0


def read_csv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        rows = np.array(
            [[float(v) for v in line.strip().split(",")] for line in f if line.strip()]
        )
    return rows, idx


def slerp(q0, q1, u):
    """四元数球面插值。q0/q1 为 (...,4) 的 xyzw 数组，u 标量或数组。"""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)  # 符号对齐
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-8  # 近平行：退化为线性插值
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


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    gyro_path, pose_path = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(gyro_path))
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    g, gi = read_csv(gyro_path)
    p, pi = read_csv(pose_path)
    t_g = g[:, gi["time"]]
    q_g = np.stack([g[:, gi[c]] for c in ("qx", "qy", "qz", "qw")], axis=1)
    t_p = p[:, pi["time"]]
    q_p = np.stack([p[:, pi[c]] for c in ("qx", "qy", "qz", "qw")], axis=1)

    # pose_cov SLERP 到 IMU 时刻：找相邻两帧，前不足处保持第一帧
    j = np.clip(np.searchsorted(t_p, t_g) - 1, 0, len(t_p) - 2)
    span = t_p[j + 1] - t_p[j]
    u = np.where(span > 0, (t_g - t_p[j]) / np.where(span > 0, span, 1.0), 0.0)
    u = np.clip(u, 0.0, 1.0)
    q_ref = slerp(q_p[j], q_p[j + 1], u[:, None])

    # 测地角误差（取 |dot|，q 与 -q 同一旋转）
    dots = np.abs(np.sum(q_g * q_ref, axis=1))
    err_deg = np.degrees(2 * np.arccos(np.clip(dots, -1.0, 1.0)))

    n = len(err_deg)
    print(f"帧数: {n}（gyro {n}，pose_cov {len(t_p)}）")
    print(f"测地角误差: 均值 {err_deg.mean():.3f}°  中位数 {np.median(err_deg):.3f}°  "
          f"95 分位 {np.percentile(err_deg, 95):.3f}°  最大 {err_deg.max():.3f}°")
    print(f"全程 < {THRESHOLD_DEG}° 的帧占比: {100 * np.mean(err_deg < THRESHOLD_DEG):.2f}%")
    print(f"初始姿态（积分第 0 帧）: roll {quat_xyzw_to_rpy_deg(q_g[:1])[0, 0]:+.3f}°  "
          f"pitch {quat_xyzw_to_rpy_deg(q_g[:1])[0, 1]:+.3f}°（参考 −0.197° / +0.546°）")

    fig_dir = os.path.join(out_dir, "figures")

    # 图 1：误差曲线
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_g - t_g[0], err_deg, lw=0.6)
    ax.axhline(THRESHOLD_DEG, color="r", ls="--", lw=1, label=f"{THRESHOLD_DEG}° 验收线")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("测地角误差 (°)")
    ax.set_title("M2 纯陀螺积分 vs FAST-LIO 姿态")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "05_gyro_error.png"), dpi=150)

    # 图 2：RPY 三方对比（积分 / 观测）
    rpy_g = quat_xyzw_to_rpy_deg(q_g)
    rpy_p = quat_xyzw_to_rpy_deg(q_ref)
    names = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(t_g - t_g[0], rpy_g[:, k], lw=0.8, label="纯陀螺积分")
        ax.plot(t_g - t_g[0], rpy_p[:, k], lw=0.8, ls="--", label="FAST-LIO（SLERP 对齐）")
        ax.set_ylabel(f"{names[k]} (°)")
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("M2：纯陀螺积分与观测姿态对比")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "06_gyro_vs_pose.png"), dpi=150)
    print(f"图已存: {fig_dir}/05_gyro_error.png, 06_gyro_vs_pose.png")

    return 0 if err_deg.max() < THRESHOLD_DEG else 1


if __name__ == "__main__":
    sys.exit(main())
