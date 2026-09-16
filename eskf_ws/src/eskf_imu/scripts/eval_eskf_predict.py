#!/usr/bin/env python3
"""M3 验收：关掉观测的 ESKF 预测步 vs M2 纯陀螺积分，逐帧一致性 + P 对角线检查。

用法（不依赖 ROS）：
    python3 scripts/eval_eskf_predict.py output/ekf_out.csv output/gyro_out.csv output

两个节点吃同一份 /imu、用同一套初始化（前 1000 帧估计 b_g 和初始姿态）、姿态推进
公式完全相同，所以两份 CSV 应当逐行对应（帧数相同、时间戳相同）。验收标准：

1. 姿态逐帧一致：四元数分量差最大 < 1e-6（预测步就是"带协方差的陀螺积分"）
2. P 对角线只增不减：零偏块严格单调；姿态块放 1% 相对容差（持续旋转段 F 左上块的
   (I − [ω×]Δt) 会把互协方差 P_θb 在机体系里转一下，个别体轴方差会小幅下降，
   是坐标效应不是 bug —— 详见脚本内注释）

不通过时按数值大小提示排查方向（见 plan.md 陷阱清单）。
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# matplotlib 认不出 Noto CJK 的 .ttc 集合，需手动注册字体文件（与 eval_gyro.py 同法），
# 否则中文标题会画成方框
from pathlib import Path

import matplotlib.font_manager as fm

_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_NOTO).exists():
    fm.fontManager.addfont(_NOTO)  # 只注册第一张脸（JP），字形覆盖含简体
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np

TOL = 1e-6  # plan.md M3 验收：逐帧一致（四元数分量差）
REL_DIP_TOL = 0.01  # 姿态块 P 对角线允许的最大相对降幅（旋转段坐标效应，见验收 2 注释）


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
    fig_dir = os.path.join(out_dir, "figures")  # 与 M1/M2 一致：图统一放 figures/
    os.makedirs(fig_dir, exist_ok=True)

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

    # ---- 验收 2：P 对角线只增不减 ----
    # 零偏块（p33..p55）必须严格单调：随机游走每秒固定入账，没有任何机制让它降。
    # 姿态块（p00..p22）在持续旋转段可能出现 *小幅* 下降：F 的左上块 (I − [ω×]Δt)
    # 会把互协方差 P_θb 在机体系里"转"一下，而 -Δt(P_θb + P_θbᵀ) 对某个体轴方差的
    # 贡献会随旋转翻号。这是姿态误差定义在机体系导致的坐标效应 + 一阶离散化近似，
    # 不是 bug（本项目实测：73.7–98.7 s 绕圈段 p00/p11 下降 <1%，p22 与零偏块始终单调）。
    # 所以姿态块放一个相对容差，只把"真正在缩水"判为失败。
    p_names = ["p00", "p11", "p22", "p33", "p44", "p55"]
    if not set(p_names) <= set(ekf.dtype.names):
        sys.exit("ekf_out.csv 缺少 P 对角线列（p00..p55）")
    P = np.column_stack([ekf[c] for c in p_names])
    if P[0].max() < 1e-12:
        sys.exit("P 对角线全为 0：CSV 输出精度不够（早期版本用 fixed 默认 6 位，"
                 "P0=1e-8 会被写成 0）。用当前 eskf_node 重跑回放即可")

    print(f"\n[验收 2] P 对角线只增不减（零偏块严格单调；姿态块相对容差 {REL_DIP_TOL:.0%}）")
    bad = []
    for k, name in enumerate(p_names):
        d = np.diff(P[:, k])
        i = int(d.argmin())
        dip = -d[i]
        rel = dip / P[i, k] if P[i, k] > 0 else 0.0
        strict = k >= 3  # 前三个是姿态块（可放宽），后三个是零偏块（必须严格单调）
        ok = (dip <= 0) if strict else (rel <= REL_DIP_TOL)
        if not ok:
            bad.append(name)
        note = ""
        if dip > 0:
            note = f"（旋转段坐标效应，相对 {rel:.2%}，可接受）" if not strict else ""
        print(f"  {name}: {P[0, k]:.3e} → {P[-1, k]:.3e}  最大降幅 {dip:.3e} "
              f"（相对 {rel:.2%}）  {'✅' if ok else '❌'}{note}")

    if bad:
        sys.exit(f"P 对角线出现异常下降：{bad}。排查方向：\n"
                 "  - Q 是否漏加（漏了则 P 完全不涨）\n"
                 "  - 零偏块下降 → 随机游走项 σ_bg²Δt 漏了\n"
                 "  - 姿态块下降远超 1% → F 的互协方差块 -Δt·I 符号写反")

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
    fig.savefig(f"{fig_dir}/07_eskf_predict_P.png", dpi=150)
    print(f"\nP 对角线演化图已保存: {out_dir}/07_eskf_predict_P.png")
    print("观察点：姿态块 60 s 后（运动段）增长明显加快；零偏块近似线性爬升（随机游走）")


if __name__ == "__main__":
    main()
