#!/usr/bin/env python3
"""M3 验收：关掉观测的 ESKF 预测步 vs M2 纯陀螺积分，逐帧一致性 + P 对角线检查。

用法（不依赖 ROS）：
    python3 scripts/eval_eskf_predict.py output/ekf_out.csv output/gyro_out.csv output

两个节点吃同一份 /imu、用同一套初始化（前 1000 帧估计 b_g 和初始姿态）、姿态推进
公式完全相同，所以两份 CSV 应当逐行对应（帧数相同、时间戳相同）。验收标准：

1. 姿态逐帧一致：四元数分量差最大 < 1e-6（预测步就是"带协方差的陀螺积分"）
2. P 对角线单调增长：无观测修正时不确定性只增不减（浮点容差内）

不通过时按数值大小提示排查方向（见 plan.md 陷阱清单）。
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOL = 1e-6  # plan.md M3 验收：逐帧一致（四元数分量差）


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {"time", "qw", "qx", "qy", "qz"}
    missing = required - set(data.dtype.names)
    if missing:
        sys.exit(f"{path}: 缺少列 {sorted(missing)}，确认文件没给错（gyro_out 与 ekf_out 都应有四元数列）")
    return data


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ekf_path, gyro_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    ekf = load_csv(ekf_path)
    gyro = load_csv(gyro_path)

    print(f"ekf_out.csv: {len(ekf)} 帧 | gyro_out.csv: {len(gyro)} 帧")
    if len(ekf) != len(gyro):
        sys.exit("帧数不同：两个节点参数（bias_window_frames）或数据不一致，先对齐再比")

    t_diff = np.abs(ekf["time"] - gyro["time"]).max()
    if t_diff > 1e-9:
        sys.exit(f"时间戳不逐帧对应（最大差 {t_diff:.3e} s）：两份 CSV 不是同一次数据跑出来的")

    # ---- 验收 1：姿态逐帧一致 ----
    q_ekf = np.column_stack([ekf[c] for c in ("qw", "qx", "qy", "qz")])
    q_gyro = np.column_stack([gyro[c] for c in ("qw", "qx", "qy", "qz")])
    comp_diff = np.abs(q_ekf - q_gyro)
    max_comp = comp_diff.max()
    max_frame = int(comp_diff.max(axis=1).argmax())
    # 测地角（度）：2*arccos(|q1·q2|)，与分量差互为印证
    dots = np.clip(np.abs((q_ekf * q_gyro).sum(axis=1)), 0.0, 1.0)
    geo_deg = np.degrees(2.0 * np.arccos(dots))

    print(f"\n[验收 1] 姿态逐帧一致性（容差 {TOL:g}）")
    print(f"  四元数分量差最大值 = {max_comp:.3e}（第 {max_frame} 帧）")
    print(f"  测地角误差最大值   = {geo_deg.max():.3e} °")
    if max_comp < TOL:
        print("  ✅ 通过：预测步与纯陀螺积分逐帧一致")
    else:
        print("  ❌ 未通过：两边的姿态推进不等价。排查方向：")
        print("    - eskf_node 的 b_g / 初始姿态初始化是否与 gyro_integral_node 相同")
        print("    - dt 跳过规则是否一致（dt<=0 两边都应跳过）")
        print("    - q ← q ⊗ Exp(ωΔt) 是否左乘写成了右乘")
        sys.exit(1)

    # ---- 验收 2：P 对角线单调增长 ----
    p_names = ["p00", "p11", "p22", "p33", "p44", "p55"]
    if not set(p_names) <= set(ekf.dtype.names):
        sys.exit("ekf_out.csv 缺少 P 对角线列（p00..p55）")
    P = np.column_stack([ekf[c] for c in p_names])
    drops = np.diff(P, axis=0).min(axis=0)  # 每个对角元在整段时间里的最大降幅
    print("\n[验收 2] P 对角线单调性（浮点容差 -1e-15）")
    for name, d, v0, v1 in zip(p_names, drops, P[0], P[-1]):
        ok = d > -1e-15
        print(f"  {name}: {v0:.3e} → {v1:.3e}  最大降幅 {d:.3e}  {'✅' if ok else '❌'}")
    if not (drops > -1e-15).all():
        sys.exit("P 对角线出现下降：检查 Q 是否漏加、F 是否写错")

    # ---- 出图：P 对角线演化 ----
    t = ekf["time"]
    labels = ["P_θx", "P_θy", "P_θz", "P_bgx", "P_bgy", "P_bgz"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        axes[0].plot(t, P[:, i], label=labels[i])
        axes[1].plot(t, P[:, 3 + i], label=labels[3 + i])
    for ax, ttl in zip(axes, ["姿态误差方差 P_θθ（rad²）", "零偏误差方差 P_bg·bg（(rad/s)²）"]):
        ax.set_yscale("log")
        ax.set_ylabel("方差（log 尺度）")
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[1].set_xlabel("t (s)")
    fig.suptitle("M3：无观测时协方差 P 对角线单调增长（log 尺度）")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/07_eskf_predict_P.png", dpi=150)
    print(f"\nP 对角线演化图已保存: {out_dir}/07_eskf_predict_P.png")
    print("观察点：姿态块 60 s 后（运动段）增长明显加快；零偏块近似线性爬升（随机游走）")


if __name__ == "__main__":
    main()
