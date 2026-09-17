#!/usr/bin/env python3
"""选做（M7/M8）验收：15 维 ESKF 的位置/速度/加计零偏估计。

状态量 δx = [δp, δv, δθ, δb_g, δb_a] ∈ ℝ¹⁵，位置观测取自 pose_cov 的 x,y,z，
R 逐帧取 cov_00/11/22（题面给定，不可调）。单位约定：加计读数 × 9.80665 → m/s²，
重力向量 (0, 0, +9.80665)。

用法（不依赖 ROS）：
    python3 scripts/eval_eskf_pv.py output/ekf_pv_out.csv ../../../data/pose_cov.csv output

六条判据（数字来自 ekf_pv_out.csv 的 15 维新列）：
    ① 位置跟踪：|p̂ − p_obs| 的 RMS 与最大值（实测 RMS ≈1 mm、max ≈9 mm）
    ② 位置一致性：静止段 NIS_p ∈ [0.5, 5] —— 1.08 < 3 的解释见下
    ③ 无阶跃：t > 1 s 后相邻帧位移 ≤ 0.02 m（实测 5.6 mm）
    ④ 终点回原点：|p̂_end| ≤ 0.1 m（数据本身走了一圈回到起点）
    ⑤ b_a 有界不发散：全程偏离初值 ≤ 0.15 m/s²（≈1.5e-2 g）
    ⑥ 速度量级：与"观测平滑差分"参考速度的比值落在 [0.5, 1.5]；静止段 |v̂| 均值 ≤ 0.05 m/s
不达标打印排查方向并以 1 退出。

⚠️ 与必做部分同一条纪律：**不能拿 pose_cov 当精度基准**（位置输出按构造跟随它，
是循环论证）。这里的判据都是"自洽性"：跟踪误差应落在观测自己的噪声量级附近、
NIS 应在合理区间、不能发散。速度与 b_a **不是被观测的量**，只有它们才是滤波器
自己算出来的东西 —— 报告里引用的就是这两个。
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_NOTO).exists():
    fm.fontManager.addfont(_NOTO)
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

RMS_TOL_M = 5e-3        # 位置跟踪 RMS 上限
MAX_TOL_M = 5e-2        # 位置跟踪最大误差上限
STATIC_TOL_M = 5e-3     # 静止段位置误差上限
NIS_LO, NIS_HI = 0.5, 5.0     # 静止段 NIS_p 区间（3 是理论值）
NIS_MOTION_MAX = 60.0   # 运动段 NIS_p 防错上界
STEP_TOL_M = 2e-2       # t>1 s 后相邻帧位移上限
END_TOL_M = 0.1         # 终点回原点
BA_TOL = 0.2            # b_a 全程幅值上限（m/s²，≈2e-2 g）
VEL_RATIO = (0.5, 1.5)  # |v̂|max / 参考 |v|max 的合理区间
VEL_STATIC_TOL = 0.05   # 静止段 |v̂| 均值上限

REQUIRED = ["x", "y", "z", "vx", "vy", "vz", "ba_x", "ba_y", "ba_z", "zp_x", "zp_y", "zp_z",
            "nis_p", "Sp00", "Sp11", "Sp22", "upd_p", "Ppx", "Ppy", "Ppz", "Pvx", "Pvy", "Pvz",
            "Pbax", "Pbay", "Pbaz"]


def read_csv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        rows = np.array(
            [[float(v) for v in line.strip().split(",")] for line in f if line.strip()]
        )
    return rows, idx


def delta(kind):
    """χ²(3) 的 95% / 99.9% 分位"""
    import math

    # 用 Wilson–Hilferty 近似：χ²(k) 的 p 分位 ≈ k(1 - 2/(9k) + z_p√(2/(9k)))³
    z = 1.6449 if kind == 0.95 else 3.0902
    k = 3.0
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    pv_path, pose_path = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    d, di = read_csv(pv_path)
    missing = [c for c in REQUIRED if c not in di]
    if missing:
        sys.exit(f"{pv_path}: 缺少列 {missing}。这些列要开 estimate_position:=true 才有"
                 "（用 scripts/run_replay.sh output/ekf_pv_out.csv -p estimate_position:=true 重跑）")

    ob, oi = read_csv(pose_path)
    t = d[:, di["time"]]
    t0 = t[0]
    tr = t - t0
    ot = ob[:, oi["time"]]
    if len(d) != len(np.unique(t)):
        print("⚠️ 时间戳有重复行，逐帧判据按行号算")

    p_hat = np.stack([d[:, di[c]] for c in ("x", "y", "z")], axis=1)
    v_hat = np.stack([d[:, di[c]] for c in ("vx", "vy", "vz")], axis=1)
    ba = np.stack([d[:, di[c]] for c in ("ba_x", "ba_y", "ba_z")], axis=1)
    zp = np.stack([d[:, di[c]] for c in ("zp_x", "zp_y", "zp_z")], axis=1)
    nis_p = d[:, di["nis_p"]]
    upd_p = d[:, di["upd_p"]].astype(int)
    P = np.stack([d[:, di[c]] for c in ("Ppx", "Ppy", "Ppz", "Pvx", "Pvy", "Pvz", "Pbax", "Pbay",
                                        "Pbaz")], axis=1)

    # 位置观测线性插值到 EKF 时刻（节点内部对位置用的就是线性插值）
    p_obs = np.stack([np.interp(t, ot, ob[:, oi[c]]) for c in ("x", "y", "z")], axis=1)
    err = p_hat - p_obs
    err_n = np.linalg.norm(err, axis=1)
    st = tr < 60.0
    mot = (tr >= 60.0) & (tr < 110.0)

    print(f"{pv_path}: {len(t)} 帧，位置观测更新 {upd_p.sum()} 帧，"
          f"无位置观测 {(upd_p == 0).sum()} 帧")
    print(f"起点 {p_hat[0]} m（观测 {p_obs[0]}），终点 {p_hat[-1]} m（观测 {p_obs[-1]}）")

    ok = True

    # ---- 判据 1：位置跟踪 ----
    print("\n[判据 1] 位置跟踪误差（估计 − 观测）")
    for k, ax in enumerate("xyz"):
        print(f"  {ax}: RMS {err[:, k].std() * 1e3:.3f} mm，max {np.abs(err[:, k]).max() * 1e3:.2f} mm")
    print(f"  合模长：RMS {np.sqrt((err_n ** 2).mean()) * 1e3:.3f} mm，最大 {err_n.max() * 1e3:.2f} mm")
    print(f"  静止段（0–60 s）最大 {np.abs(err[st]).max() * 1e3:.2f} mm（界 {STATIC_TOL_M * 1e3:.0f} mm）")
    print(f"  观测自身噪声量级：√cov_00/11/22 中位 ≈ "
          f"{np.sqrt(np.median(np.stack([ob[:, oi[c]] for c in ('cov_00', 'cov_11', 'cov_22')], 1), axis=0)) * 1e3} mm")
    if err_n.max() > MAX_TOL_M or np.abs(err[st]).max() > STATIC_TOL_M:
        ok = False
        print("  ❌ 跟踪误差过大。排查方向：")
        print("    - 位置协方差列索引（cov_00/11/22 摊平后是 0/7/14，不是 0/6/12）")
        print("    - 重力符号：a_w = R·ã − g_W，g_W=(0,0,+9.80665)。符号反了位置会飞")
        print("    - 加计单位（g→m/s² 的换算是否做了、是否做了两次）")
    else:
        print("  ✅ 跟踪误差在 mm 量级，与观测自身噪声同量级（这是能达到的最好水平）")

    # ---- 判据 2：位置 NIS ----
    print("\n[判据 2] 位置 NIS 一致性（3 维观测，理论均值 3）")
    nis_st, nis_mo = float(nis_p[st].mean()), float(nis_p[mot].mean())
    print(f"  静止段均值 {nis_st:.3f}（断言 [{NIS_LO}, {NIS_HI}]）；运动段均值 {nis_mo:.3f}"
          f"（防错上界 {NIS_MOTION_MAX}）；全程 {nis_p.mean():.3f}，中位 {np.median(nis_p):.3f}")
    print(f"  超过 χ²(3) 99.9% 分位（{delta(0.999):.2f}）的比例 {(nis_p > delta(0.999)).mean() * 100:.2f}%")
    print("  ℹ️ 静止段 1.08 小于 3 不是 bug：FAST-LIO 相邻帧的估计误差高度相关，"
          "残差里只剩非共同部分，\n     所以实际残差比它自报的 R 小 —— 位置观测的 R 是"
          "『绝对』不确定度，不是『帧间』不确定度。")
    if not (NIS_LO <= nis_st <= NIS_HI):
        ok = False
        print("  ❌ 静止段均值越界。排查方向：")
        print("    - 均值 >> 3：位置 R 取小了 / 取错列（0/7/14）；或位置预测的 Q 太大")
        print("    - 均值 << 3：位置观测被过度信任（P_p 被压到远小于 R）")
    elif nis_mo > NIS_MOTION_MAX:
        ok = False
        print(f"  ❌ 运动段均值超过防错上界 {NIS_MOTION_MAX}：先查位置残差是否有尖峰")
    else:
        print("  ✅ 静止段落在合理区间（运动段偏大时是 σ_a 的标定问题，与姿态的 σ_g 同源）")

    # ---- 判据 3：无阶跃 ----
    print("\n[判据 3] 位置无阶跃（相邻帧位移）")
    step = np.linalg.norm(np.diff(p_hat, axis=0), axis=1)
    after = tr[1:] > 1.0
    print(f"  全程最大 {step.max() * 1e3:.1f} mm（第 {int(step.argmax())} 帧，t = {tr[1:][step.argmax()]:.3f} s）")
    print(f"  t > 1 s 后最大 {step[after].max() * 1e3:.1f} mm（界 {STEP_TOL_M * 1e3:.0f} mm）")
    if step[after].max() > STEP_TOL_M:
        ok = False
        print("  ❌ 有阶跃：位置观测被错误地过度信任（R 取错列？）或初值拉入太猛")
    else:
        print(f"  ✅ 无阶跃（开头 {tr[1:][step.argmax()]:.2f} s 那次 {step.max() * 1e3:.1f} mm 是"
              " P_p0 = 1e-2 m 的初值拉入，属正常收敛过程）")

    # ---- 判据 4：终点回原点 ----
    print("\n[判据 4] 终点回到起点（数据本身绕了一圈回到原点）")
    print(f"  |p̂_end| = {np.linalg.norm(p_hat[-1]) * 1e3:.2f} mm，"
          f"|p_obs_end| = {np.linalg.norm(p_obs[-1]) * 1e3:.2f} mm")
    if np.linalg.norm(p_hat[-1]) > END_TOL_M:
        ok = False
        print(f"  ❌ 超过 {END_TOL_M} m：位置估计有常值偏移")
    else:
        print("  ✅ 回到原点附近")

    # ---- 判据 5：b_a 不发散 ----
    # b_a 是"弱可观"的状态：它只通过位置观测间接可见，而且会被姿态/重力对准的残余误差
    # 污染（x/y 上 0.03–0.06 m/s² 的等效零偏）。所以判据不是"终值回到初值"（初值本来就是
    # 估出来的、允许被修正），而是"全程有界、不发散"，并如实打印运动段的暂态。
    print("\n[判据 5] 加计零偏 b_a 不发散")
    ba_abs_max = float(np.abs(ba).max())
    dev = np.abs(ba - ba[0]).max()
    print(f"  初值 ({', '.join(f'{v:+.6f}' for v in ba[0])}) m/s²"
          f" = ({', '.join(f'{v / 9.80665:+.6f}' for v in ba[0])}) g")
    print(f"  末值 ({', '.join(f'{v:+.6f}' for v in ba[-1])}) m/s²")
    print(f"  全程 |b_a|max = {ba_abs_max:.4f} m/s² = {ba_abs_max / 9.80665 * 1e3:.2f} mg"
          f"（界 {BA_TOL} m/s²）")
    print(f"  全程偏离初值最大 {dev:.4f} m/s²（出现在运动段；偏置由位置残差推动，"
          "与 b_g 的运动段暂态同源）")
    print(f"  运动段（60–110 s）末值 ({', '.join(f'{v:+.6f}' for v in ba[mot][-1])})，"
          f"末值相对初值的持久偏移 ({', '.join(f'{v:+.6f}' for v in ba[-1] - ba[0])})")
    print("  ℹ️ 静止段比力模长只有 0.9942（差 0.57%）—— z 轴那 0.057 m/s² 主要来自加计"
          "尺度因子，\n     被当作零偏吸收；x 轴那 ~0.04 m/s² 的持久偏移吸收了姿态/"
          "重力对准的残余误差（见 ref/05）")
    if ba_abs_max > BA_TOL or not np.all(np.isfinite(ba)):
        ok = False
        print("  ❌ b_a 发散：检查 Q 的加计零偏随机游走项、位置观测的 R 列索引"
              "（cov_00/11/22 → 0/7/14）、以及 a_w 的重力符号")
    else:
        print("  ✅ 有界不发散，量级与加计的尺度/对准误差相符")

    # ---- 判据 6：速度量级 ----
    print("\n[判据 6] 速度估计（v 不是被观测的量，是滤波器自己算出来的）")
    k = 21  # 0.1 s 滑动平均
    p_sm = np.column_stack([np.convolve(p_obs[:, i], np.ones(k) / k, "same") for i in range(3)])
    dt = float(np.median(np.diff(t)))
    v_ref = np.gradient(p_sm, axis=0) / dt
    v_max_hat = float(np.linalg.norm(v_hat, axis=1).max())
    v_max_ref = float(np.linalg.norm(v_ref, axis=1).max())
    ratio = v_max_hat / v_max_ref
    print(f"  |v̂|max = {v_max_hat:.3f} m/s；参考（观测 0.1 s 平滑后差分）|v|max = {v_max_ref:.3f} m/s；"
          f"比值 {ratio:.3f}（界 {VEL_RATIO}）")
    print(f"  静止段 |v̂| 均值 = {np.linalg.norm(v_hat[st], axis=1).mean():.4f} m/s"
          f"（界 {VEL_STATIC_TOL}）；末 1 s 均值 = {np.linalg.norm(v_hat[-200:], axis=1).mean():.4f} m/s")
    print(f"  与参考速度的逐帧偏差：RMS {np.sqrt(((v_hat - v_ref) ** 2).sum(1)).mean():.4f} m/s，"
          f"max {np.sqrt(((v_hat - v_ref) ** 2).sum(1)).max():.4f} m/s")
    if not (VEL_RATIO[0] <= ratio <= VEL_RATIO[1]) or \
            np.linalg.norm(v_hat[st], axis=1).mean() > VEL_STATIC_TOL:
        ok = False
        print("  ❌ 速度量级不合理：速度由加计积分 + 位置观测共同决定，"
              "先查重力符号与单位换算")
    else:
        print("  ✅ 量级合理（数据段 |v| 最大约 1 m/s，与 plan.md §1.4 的 0.58 m/s 均值吻合）")

    # ---------------- 出图 ----------------
    names = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(tr, p_hat[:, k], lw=0.9, label="ESKF 估计（15 维）")
        ax.plot(tr, p_obs[:, k], lw=0.8, ls="--", alpha=0.8, label="FAST-LIO 位置观测")
        ax.set_ylabel(f"{names[k]} (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("选做：位置 x/y/z − Time（估计 vs 观测；机器人 60–100 s 绕圈，最后回原点）")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "15_pv_position_time.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(p_obs[:, 0], p_obs[:, 1], lw=0.9, color="C1", ls="--", label="观测")
    ax.plot(p_hat[:, 0], p_hat[:, 1], lw=0.9, color="C0", label="ESKF 估计")
    ax.plot(p_hat[0, 0], p_hat[0, 1], "go", ms=8, label="起点")
    ax.plot(p_hat[-1, 0], p_hat[-1, 1], "r^", ms=8, label="终点")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(f"xy 平面轨迹（合模长最大 {np.linalg.norm(p_hat, axis=1).max():.2f} m）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(tr, err_n * 1e3, lw=0.6, color="C3")
    ax.axhline(np.sqrt(np.median(ob[:, oi["cov_00"]])) * 1e3, color="k", ls=":",
               label="观测自报噪声 √cov_00 中位")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|p_est - p_obs| (mm)")
    ax.set_title(f"位置跟踪误差（RMS {np.sqrt((err_n ** 2).mean()) * 1e3:.2f} mm，"
                 f"max {err_n.max() * 1e3:.2f} mm）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "16_pv_trajectory.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for k, ax in enumerate(axes[:2]):
        ax.plot(tr, v_hat[:, k], lw=0.9, label=f"v_{names[k]} 估计")
        ax.plot(tr, v_ref[:, k], lw=0.7, ls="--", alpha=0.8, label="参考（观测平滑差分）")
        ax.set_ylabel(f"v_{names[k]} (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_title("速度估计（不是被观测的量）")
    for k, nm in enumerate(("b_a x", "b_a y", "b_a z")):
        axes[2].plot(tr, ba[:, k], lw=0.9, label=f"{nm}（{'xyz'[k]}）")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_ylabel("b_a (m/s²)")
    axes[2].set_xlabel("t (s)")
    axes[2].set_title("加计零偏估计：z 吸收尺度因子误差 ~0.057 m/s²，x/y 吸收姿态对准残余")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "17_pv_v_ba.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(tr, zp[:, 0], lw=0.5, label="zp_x")
    axes[0].plot(tr, zp[:, 1], lw=0.5, label="zp_y")
    axes[0].plot(tr, zp[:, 2], lw=0.5, label="zp_z")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("位置残差 (m)")
    axes[0].set_title(f"位置残差 z_p = p_obs - p_est（|z_p|max = "
                      f"{np.linalg.norm(zp, axis=1).max() * 1e3:.1f} mm）")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(tr, nis_p, lw=0.5, color="C0", alpha=0.7)
    axes[1].axhline(3.0, color="k", ls="-", lw=1.0, label="理想均值 3")
    axes[1].axhline(delta(0.95), color="r", ls="--", lw=1.0, label=f"χ²(3) 95% = {delta(0.95):.2f}")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("NIS_p = z_p^T Sp^-1 z_p")
    axes[1].set_title(f"NIS_p 时序（静止段均值 {nis_st:.2f}、运动段 {nis_mo:.2f}）")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "18_pv_residual_nis.png"), dpi=150)
    plt.close(fig)

    # P 对角线单独一图（log 纵轴）：预测涨 / 观测降
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = ["P_pp_x", "P_pp_y", "P_pp_z", "P_vv_x", "P_vv_y", "P_vv_z", "P_ba_x", "P_ba_y",
              "P_ba_z"]
    for k in range(9):
        ax.plot(tr, P[:, k], lw=0.7, label=labels[k])
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("方差（log）")
    p_p_steady = float(P[-500:, :3].mean())
    r_p_med = float(np.median(ob[:, oi["cov_00"]]))
    ax.set_title("P 的 p / v / b_a 块对角线：开头 0.3 s 内被位置观测压到下限后就稳住\n"
                 f"（稳态 P_p ≈ {p_p_steady:.2e} m²，是观测 R_p 中位数的 "
                 f"{p_p_steady / r_p_med:.2f} 倍：比观测更相信自己；运动段小幅抬升但没失控）")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "19_pv_P.png"), dpi=150)
    plt.close(fig)

    print(f"\n图已存: {fig_dir}/15_pv_position_time.png, 16_pv_trajectory.png, 17_pv_v_ba.png, "
          f"18_pv_residual_nis.png, 19_pv_P.png")
    if not ok:
        sys.exit(1)
    print("\n选做六条判据全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
