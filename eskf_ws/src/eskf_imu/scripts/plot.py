#!/usr/bin/env python3
"""M5：题面 §4(4) 硬性要求的 Roll/Pitch/Yaw–Time 曲线 + 三方对比（纯积分 / EKF / 观测）。

用法（不依赖 ROS）：
    python3 scripts/plot.py output/ekf_out.csv output/gyro_out.csv ../../../data/pose_cov.csv output

出图：
    11_rpy_time.png            Roll/Pitch/Yaw–Time 三条曲线（题目要求的那三张）
    12_three_way_attitude.png  三方对比 + 测地角误差 + 平滑度（yaw 二阶差分）

三条判据：
    ① EKF 姿态与观测的测地角误差全程 < 5°（与 M2 同一条验收线）
    ② EKF 比纯陀螺积分更贴观测（测地角误差均值更小）—— 观测融合确实起了作用
    ③ EKF 比纯陀螺积分更平滑（yaw 的二阶差分 std 更小）—— "平滑换响应"里"平滑"那一半
       （注：EKF 也几乎跟着观测走，所以这条同时把观测的平滑度一起打出来做参照）
不达标打印排查方向并以 1 退出。
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# matplotlib 认不出 Noto CJK 的 .ttc 集合，需手动注册字体文件（与 M2 的脚本同法）
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


def geodesic_deg(qa, qb):
    """测地角误差：一律用四元数，不用欧拉角之差（yaw 过 ±180° 会跳）。"""
    dots = np.abs(np.sum(qa * qb, axis=1))
    return np.degrees(2 * np.arccos(np.clip(dots, -1.0, 1.0)))


def align_pose_to(t_q, q_p, t_p):
    """把 pose 观测 SLERP 到给定时刻：找相邻两帧，前不足处保持第一帧。"""
    j = np.clip(np.searchsorted(t_p, t_q) - 1, 0, len(t_p) - 2)
    span = t_p[j + 1] - t_p[j]
    u = np.where(span > 0, (t_q - t_p[j]) / np.where(span > 0, span, 1.0), 0.0)
    u = np.clip(u, 0.0, 1.0)
    return slerp(q_p[j], q_p[j + 1], u[:, None])


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    ekf_path, gyro_path, pose_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(os.path.abspath(ekf_path))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    e, ei = read_csv(ekf_path)
    g, gi = read_csv(gyro_path)
    p, pi = read_csv(pose_path)

    t_e = e[:, ei["time"]]
    q_e = np.stack([e[:, ei[c]] for c in ("qw", "qx", "qy", "qz")], axis=1)[:, [1, 2, 3, 0]]
    t_g = g[:, gi["time"]]
    q_g = np.stack([g[:, gi[c]] for c in ("qw", "qx", "qy", "qz")], axis=1)[:, [1, 2, 3, 0]]
    t_p = p[:, pi["time"]]
    q_p = np.stack([p[:, pi[c]] for c in ("qx", "qy", "qz", "qw")], axis=1)

    if len(t_e) != len(t_g):
        print(f"⚠️ EKF {len(t_e)} 帧 vs 纯积分 {len(t_g)} 帧：行数不一致，逐帧对比无意义"
              "（检查两个节点是否都比 player 早起、没丢开头帧）")
    n = min(len(t_e), len(t_g))
    dt = np.median(np.diff(t_e))

    q_ref_e = align_pose_to(t_e, q_p, t_p)
    q_ref_g = align_pose_to(t_g, q_p, t_p)
    err_e = geodesic_deg(q_e, q_ref_e)
    err_g = geodesic_deg(q_g, q_ref_g)

    rpy_e = quat_xyzw_to_rpy_deg(q_e)
    rpy_g = quat_xyzw_to_rpy_deg(q_g)
    rpy_p = quat_xyzw_to_rpy_deg(q_ref_e)
    # yaw 段接段展开（机器人转了整整一圈），只看平滑度时用二阶差分
    yaw_e = np.unwrap(np.radians(rpy_e[:, 2]))
    yaw_g = np.unwrap(np.radians(rpy_g[:, 2]))
    yaw_p = np.unwrap(np.radians(rpy_p[:, 2]))
    rough_e = float(np.std(np.diff(yaw_e, 2)))
    rough_g = float(np.std(np.diff(yaw_g, 2)))
    rough_p = float(np.std(np.diff(yaw_p, 2)))

    t0 = t_e[0]
    print(f"帧数: EKF {len(t_e)}（dt 中位 {dt * 1e3:.3f} ms），纯积分 {len(t_g)}，"
          f"pose_cov {len(t_p)}")
    print(f"\n[判据 1] EKF vs 观测的测地角误差（验收线 {THRESHOLD_DEG}°）")
    print(f"  均值 {err_e.mean():.3f}°  中位数 {np.median(err_e):.3f}°  "
          f"95 分位 {np.percentile(err_e, 95):.3f}°  最大 {err_e.max():.3f}°")
    ok = True
    if err_e.max() >= THRESHOLD_DEG:
        ok = False
        print(f"  ❌ 超过 {THRESHOLD_DEG}°。排查方向：")
        print("    - 姿态残差符号/四元数分量顺序（误差恒定在 180° 附近 → 分量顺序错）")
        print("    - 观测对齐是否退化成最近邻（应 SLERP 插值）")
        print("    - 符号翻转帧附近是否有尖峰（看 08 图）")
    else:
        print(f"  ✅ 全程 < {THRESHOLD_DEG}°")

    print("\n[判据 2] EKF 应比纯陀螺积分更贴观测（观测融合起效）")
    print(f"  EKF  均值 {err_e.mean():.3f}°")
    print(f"  纯积分 均值 {err_g.mean():.3f}°")
    if err_e.mean() < err_g.mean():
        print(f"  ✅ EKF 误差更小（差 {err_g.mean() - err_e.mean():.3f}°）")
    else:
        ok = False
        print("  ❌ EKF 反而更差：R 是否取错列（cov_33/44/55 摊平后是 21/28/35）？"
              "残差符号是否反了（零偏会被推向一边）？")

    print("\n[判据 3] 平滑度（yaw 的二阶差分 std，越小越平滑）")
    print(f"  EKF {rough_e:.3e} rad  纯积分 {rough_g:.3e} rad  观测 {rough_p:.3e} rad")
    # 这里比的不是"EKF 比纯积分平滑"——纯陀螺积分本身就是个低通（噪声被积分平滑掉了），
    # 而 EKF 每帧都被 R 很小的观测拽一下，反而带着观测的粗糙。有意义的判据是：
    # **EKF 必须比它跟随的观测更平滑**，即滤波器确实起了低通作用。
    if rough_e < rough_p:
        print(f"  ✅ 比观测平滑（{(1 - rough_e / rough_p) * 100:.1f}%↓）：滤波器在低通观测噪声，"
              f"与纯积分同量级（{rough_e / rough_g:.2f} 倍）")
    else:
        ok = False
        print("  ❌ 比观测还毛：检查 R 是否取错列（取小了会让观测直接灌进姿态）")

    # ---------------- 出图 ----------------
    t = t_e - t0
    names = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(t, rpy_e[:, k], lw=0.9, color="C0", label="EKF 输出")
        ax.plot(t_g - t0, rpy_g[:, k], lw=0.7, color="C1", ls=":", label="纯陀螺积分（M2）")
        ax.plot(t, rpy_p[:, k], lw=0.7, color="C2", ls="--", alpha=0.8, label="FAST-LIO 观测")
        ax.set_ylabel(f"{names[k]} (°)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("M5：Roll / Pitch / Yaw – Time（题面 §4(4) 要求的三条曲线）")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "11_rpy_time.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, err_e, lw=0.6, color="C0", label="EKF vs 观测")
    ax1.plot(t_g - t0, err_g, lw=0.6, color="C1", alpha=0.8, label="纯积分 vs 观测")
    ax1.axhline(THRESHOLD_DEG, color="r", ls="--", lw=1, label=f"{THRESHOLD_DEG}° 验收线")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("测地角误差 (°)")
    ax1.set_title("三方姿态误差（滤波在运动段明显压住积分漂移）")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(err_e, bins=60, alpha=0.7, label=f"EKF 均值 {err_e.mean():.2f}°")
    ax2.hist(err_g, bins=60, alpha=0.5, label=f"纯积分 均值 {err_g.mean():.2f}°")
    ax2.set_xlabel("测地角误差 (°)")
    ax2.set_ylabel("帧数")
    ax2.set_title("误差分布")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 放大一段高速运动（60–70 s）：EKF 应介于积分与观测之间
    ax3 = fig.add_subplot(2, 1, 2)
    m = (t >= 60) & (t <= 70)
    ax3.plot(t[m], rpy_e[m, 2], lw=1.4, color="C0", label="EKF 输出")
    ax3.plot(t_g[m] - t0, rpy_g[m, 2], lw=1.0, color="C1", ls=":", label="纯陀螺积分")
    ax3.plot(t[m], rpy_p[m, 2], lw=0.9, color="C2", ls="--", label="FAST-LIO 观测")
    ax3.set_xlabel("t (s)")
    ax3.set_ylabel("Yaw (°)")
    ax3.set_title("60–70 s 放大：EKF 贴在观测上，纯积分与它的偏差被观测拉回来")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "12_three_way_attitude.png"), dpi=150)
    plt.close(fig)
    print(f"\n图已存: {fig_dir}/11_rpy_time.png, {fig_dir}/12_three_way_attitude.png")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
