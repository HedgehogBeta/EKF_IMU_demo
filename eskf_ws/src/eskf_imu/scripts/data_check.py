#!/usr/bin/env python3
"""M1 数据体检：Δt 直方图、|a| 曲线、协方差对角项曲线、姿态速率对比。

用法: python3 data_check.py <imu.csv> <pose_cov.csv> [输出目录]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# matplotlib 认不出 Noto CJK 的 .ttc 集合，需手动注册字体文件
_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_NOTO).exists():
    fm.fontManager.addfont(_NOTO)  # 只注册第一张脸（JP），字形覆盖含简体
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_numeric_csv(path):
    # csv 模块遇 NUL 会直接抛错，先抹掉；坏行走 ValueError 被跳过
    rows, skipped = [], 0
    with open(path, newline="") as f:
        cleaned = (line.replace("\0", "") for line in f)
        reader = csv.reader(cleaned)
        header = next(reader)
        for fields in reader:
            try:
                rows.append([float(x) for x in fields])
            except ValueError:
                skipped += 1
    return header, np.asarray(rows, dtype=np.float64), skipped


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    imu_path, pose_path = sys.argv[1], sys.argv[2]
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    imu_header, imu, imu_skipped = load_numeric_csv(imu_path)
    t_imu = imu[:, imu_header.index("time")]
    acc = imu[:, [imu_header.index(c) for c in ("ax", "ay", "az")]]
    gyro = imu[:, [imu_header.index(c) for c in ("gx", "gy", "gz")]]
    dt_imu = np.diff(t_imu)

    pose_header, pose, pose_skipped = load_numeric_csv(pose_path)
    t_pose = pose[:, pose_header.index("time")]
    quat = pose[:, [pose_header.index(c) for c in ("qx", "qy", "qz", "qw")]]
    cov_diag = pose[:, [pose_header.index(f"cov_{i}{i}") for i in (3, 4, 5)]]

    # 静止段（t < 60 s）陀螺均值 = 零偏
    static = t_imu - t_imu[0] < 60.0
    bg = gyro[static].mean(axis=0)
    acc_static_mean = acc[static].mean(axis=0)
    acc_norm = np.linalg.norm(acc, axis=1)

    # 相邻四元数的测地角差分（|dot| 对 q/-q 符号翻转免疫）
    dq = np.abs(np.sum(quat[1:] * quat[:-1], axis=1))
    dtheta = 2 * np.arccos(np.clip(dq, -1.0, 1.0))
    dquat_dt = dtheta / np.diff(t_pose)
    quat_flip = np.sum(quat[1:] * quat[:-1], axis=1) < 0
    gyro_norm_deg = np.linalg.norm(gyro, axis=1) * 180 / np.pi

    print("=" * 62)
    print("M1 数据体检报告")
    print("=" * 62)
    print(f"imu.csv  有效行 {len(imu)}（跳过 {imu_skipped}），时长 {t_imu[-1]-t_imu[0]:.1f} s")
    print(f"  Δt: 均值 {dt_imu.mean()*1000:.4f} ms, 中位数 {np.median(dt_imu)*1000:.4f} ms, "
          f"最大 {dt_imu.max()*1000:.3f} ms, 最小 {dt_imu.min()*1000:.3f} ms")
    print(f"  静止段(t<60s) {static.sum()} 帧")
    print(f"  b_g 估计 = ({bg[0]:.5f}, {bg[1]:.5f}, {bg[2]:.5f}) rad/s"
          "   <- 期望 ≈ (0.0005, 0.0025, 0.0166)")
    print(f"  静止段 acc 均值 = ({acc_static_mean[0]:+.5f}, {acc_static_mean[1]:+.5f}, "
          f"{acc_static_mean[2]:+.5f}) g, 模长 {np.linalg.norm(acc_static_mean):.4f} g"
          "   <- 期望 ≈ 0.994")
    print(f"  |gyro| 最大 {gyro_norm_deg.max():.1f} °/s")
    print(f"pose_cov.csv 有效行 {len(pose)}（跳过 {pose_skipped}），起始晚 imu "
          f"{t_pose[0]-t_imu[0]:.3f} s   <- 期望 ≈ 0.170")
    print(f"  协方差对角项 cov_33/44/55: 前 {np.argmax(cov_diag.any(axis=1))} 帧含零，"
          f"非零范围 [{cov_diag[cov_diag>0].min():.2e}, {cov_diag.max():.2e}]"
          "   <- 期望 [4e-8, 2.6e-7]")
    print(f"  四元数符号翻转帧数 {quat_flip.sum()}（第 {np.argmax(quat_flip)+1} 帧起）"
          "   <- 期望 1 次")
    print(f"  四元数差分姿态速率最大 {dquat_dt.max()*180/np.pi:.1f} °/s，陀螺实测最大 "
          f"{gyro_norm_deg.max():.1f} °/s —— 差分速率偏高即时间戳抖动/混叠的证据，"
          "对齐必须插值（plan §1.5）")
    print("=" * 62)

    # 尾部大值只有个位数样本，线性轴看不见，用对数轴
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dt_imu * 1000, bins=200)
    ax.set_yscale("log")
    ax.set_xlabel("Δt (ms)")
    ax.set_ylabel("帧数（对数轴）")
    ax.set_title("IMU 相邻帧时间间隔分布（主峰 5 ms；尾部为坏行 ~10 ms 与启动抖动 ~20 ms）")
    fig.tight_layout()
    fig.savefig(out_dir / "01_dt_hist.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_imu - t_imu[0], acc_norm, lw=0.4)
    ax.axvline(60, color="r", ls="--", lw=0.8, label="t=60 s（静止段结束）")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|a| (g)")
    ax.set_title("加速度模长（静止段应 ≈ 0.994 g，单位是 g 不是 m/s²）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_acc_norm.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(("cov_33 (roll)", "cov_44 (pitch)", "cov_55 (yaw)")):
        ax.plot(t_pose - t_pose[0], cov_diag[:, i], lw=0.6, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("variance (rad²)")
    ax.set_title("姿态协方差对角项（开头 20 帧为 0，之后在 4e-8 ~ 2.6e-7 波动）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "03_cov_diag.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_pose[1:] - t_pose[0], dquat_dt * 180 / np.pi, lw=0.4,
            label="pose 四元数差分（测地角）")
    ax.plot(t_imu - t_imu[0], gyro_norm_deg, lw=0.4, alpha=0.7,
            label="|gyro|（IMU 实测）")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("角速率 (°/s)")
    ax.set_title("姿态速率：插值对齐前的体检（两者量级应一致）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04_attitude_rate.png", dpi=150)

    print(f"4 张图已存到 {out_dir}/")


if __name__ == "__main__":
    main()
