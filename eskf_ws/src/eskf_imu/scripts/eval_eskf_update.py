#!/usr/bin/env python3
"""M4 验收：ESKF 观测更新步的四项判据 + 三张图。

用法（不依赖 ROS）：
    python3 scripts/eval_eskf_update.py output/ekf_out.csv ../../../data/pose_cov.csv output

四项判据（数据来自 ekf_out.csv 里 M4 新增的 8 列）：

1. 残差无尖峰：全程 |z| 无 > 0.5 rad 的孤立尖峰；且 pose_cov 那次四元数符号翻转
   （第 17957 帧，q -> -q）附近的 |z| 不超过邻居中位数的 3 倍。
   去掉符号展开会走 SLERP 长弧、插值中点偏 180°，这里会看到 |z| ≈ 2。
2. NIS：静止段均值断言落在 [1.5, 5]（实测 2.41 ≈ 3，模型假设成立时自洽）；运动段均值只设防错上界
   并如实打印（实测 ~17，是 Q 偏小的标定问题，M5 的 Q 扫描处理）。
3. 零偏不发散：终值回到初值 1e-3 以内，且全程偏离不超过零偏本身量级。
4. 姿态无阶跃：相邻帧转角 ≤ 2°（数据真实上限 0.758°/帧），没有观测把姿态打飞。

不达标时打印排查方向（见 plan.md 陷阱清单），并以 1 退出。
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

# ---- 判据阈值 ----
MAX_Z = 0.5          # 残差绝对值上限（rad）
SPIKE_RATIO = 3.0    # 翻转帧附近 |z| 相对邻居中位数的倍数上限
NIS_LO, NIS_HI = 1.5, 5.0   # 静止段 NIS 均值区间（理想 3）
NIS_MOTION_MAX = 60.0       # 运动段 NIS 均值的防错上界（见判据 2 的说明）
BG_RECOVER = 1e-3    # 终值相对初值的最大偏离（rad/s）：plan.md 说"量级 1e-4"
BG_BOUNDED = 1.0     # 全程最大偏离 / |b_g| 的上界：不能漂到零偏本身量级
STEP_TOL_DEG = 2.0   # 相邻帧转角上限（度）
STATIC_SEC = 60.0    # 静止段时长（plan.md §1.4）
CHI2_3_95 = 7.815    # χ²(3) 的 95% 分位
CHI2_3_999 = 16.27   # χ²(3) 的 99.9% 分位

REQUIRED = ["time", "dt", "qw", "qx", "qy", "qz", "bg_x", "bg_y", "bg_z",
            "p00", "p11", "p22", "p33", "p44", "p55",
            "z_x", "z_y", "z_z", "nis", "S00", "S11", "S22", "upd"]


def chi2_3_pdf(x):
    """χ²(3) 的概率密度：x^0.5 e^{-x/2} / (2^1.5 · Γ(1.5))"""
    import math
    norm = (2.0 ** 1.5) * math.gamma(1.5)
    return np.sqrt(x) * np.exp(-x / 2.0) / norm


def find_sign_flip(q):
    """返回姿态序列里 q -> -q 翻转的样本下标（原始点积 < 0 且模长仍接近 1）。"""
    d = (q[:-1] * q[1:]).sum(axis=1)
    idx = np.where(d < 0)[0]
    return idx


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ekf_path, pose_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    fig_dir = os.path.join(out_dir, "figures")  # 与 M1/M2 一致：图统一放 figures/
    os.makedirs(fig_dir, exist_ok=True)

    data = np.genfromtxt(ekf_path, delimiter=",", names=True)
    missing = [c for c in REQUIRED if c not in data.dtype.names]
    if missing:
        sys.exit(f"{ekf_path}: 缺少列 {missing}。M4 的 ekf_out.csv 应含 z_x..upd 这 8 列"
                 "（用 M4 的 eskf_node 重跑回放即可）")

    n = len(data)
    t = data["time"]
    t0 = t[0]
    upd = np.asarray(data["upd"]).astype(int)
    z = np.column_stack([data["z_x"], data["z_y"], data["z_z"]])
    z_norm = np.linalg.norm(z, axis=1)
    nis = np.asarray(data["nis"])
    bg = np.column_stack([data["bg_x"], data["bg_y"], data["bg_z"]])
    P = np.column_stack([data[c] for c in ("p00", "p11", "p22", "p33", "p44", "p55")])
    q = np.column_stack([data[c] for c in ("qw", "qx", "qy", "qz")])

    print(f"{ekf_path}: {n} 帧 | 有观测更新 {upd.sum()} 帧 | 无观测 {(upd == 0).sum()} 帧")

    # ---- pose_cov 里那次符号翻转（本项目的关键毛刺）----
    pose = np.genfromtxt(pose_path, delimiter=",", names=True)
    q_pose = np.column_stack([pose["qw"], pose["qx"], pose["qy"], pose["qz"]])
    flips = find_sign_flip(q_pose)
    t_pose = np.asarray(pose["time"])
    if len(flips) == 0:
        print("  ⚠️ pose_cov.csv 里没检测到符号翻转，判据 1 的翻转段检查跳过")
        flip_lo = flip_hi = None
    else:
        i = int(flips[0])
        flip_lo, flip_hi = t_pose[i], t_pose[i + 1]
        rot_deg = np.degrees(2 * np.arccos(np.clip(abs(float(q_pose[i] @ q_pose[i + 1])), 0, 1)))
        print(f"  pose_cov 符号翻转：第 {i} 帧，q -> -q，真实转角仅 {rot_deg:.4f}°")
        if len(flips) > 1:
            print(f"  ⚠️ 检测到 {len(flips)} 次翻转（预期 1 次）：{flips[:5]}")

    ok = True

    # ---- 判据 1：残差无尖峰 ----
    print("\n[判据 1] 残差无尖峰")
    m = upd == 1
    if not m.any():
        sys.exit("一帧观测更新都没有：确认 pose_player_node 在跑、/pose_cov 有数据")
    z_max = z_norm[m].max()
    i_max = int(np.where(m)[0][np.argmax(z_norm[m])])
    print(f"  全程 |z| 最大值 = {z_max:.5f} rad（第 {i_max} 帧，t = {t[i_max] - t0:.2f} s）")
    if z_max >= MAX_Z:
        ok = False
        print(f"  ❌ 超过 {MAX_Z} rad。排查方向：")
        print("    - 符号对齐是否失效（翻转帧附近应看到 |z| ≈ 2 的尖峰）")
        print("    - 姿态插值是否退化成最近邻（残差被对齐误差主导）")
    else:
        print(f"  ✅ 低于 {MAX_Z} rad")

    if flip_lo is not None:
        in_win = (t >= flip_lo) & (t <= flip_hi)
        near = (~in_win) & (np.abs(t - flip_lo) < 60.0) & m
        n_win = int((in_win & m).sum())
        if n_win == 0:
            print("  ⚠️ 翻转区间内没有做更新的帧（观测缺失？），无法比对")
        else:
            med = float(np.median(z_norm[near])) if near.any() else 0.0
            peak = float(z_norm[in_win & m].max())
            ratio = peak / med if med > 0 else float("inf")
            print(f"  翻转区间（{n_win} 帧）：|z| 峰值 = {peak:.5f}，邻居（±60 s）中位数 = "
                  f"{med:.5f}，比值 {ratio:.2f}（上限 {SPIKE_RATIO}）")
            if ratio > SPIKE_RATIO:
                ok = False
                print("  ❌ 翻转帧残差相对邻居异常放大。若峰值 ≈2，就是 SLERP 走了长弧：")
                print("     检查 pose 入缓冲时的符号展开（相邻样本点积为负要整体取反）")
            else:
                print(f"  ✅ 与邻居同量级（plan.md 要求：该帧残差不应出尖峰）")

    # ---- 判据 2：NIS ----
    # 断言只压在静止段：那里模型假设全部成立（无旋转、白噪声观测），实测 NIS 均值 2.41 ≈ 3，
    # 说明更新步的 S/H/K 都对。运动段实测均值 ~17，是 Q 标定问题不是实现错误：
    #   σ_g=0.0032 取自静止段统计，运动时陀螺的尺度因子/失准/振动等未建模误差随角速率增长，
    #   Q 偏小 → 预测过度自信 → 残差相对 S 偏大。实测把 σ_g ×10 后运动段降到 1.65（但静止段
    #   掉到 0.36，变得过度保守）—— 到底取多少正是 M5 的 Q 扫描要回答的。
    # 所以 M4 只要求：静止段一致 + 运动段在防错上界内，并把运动段的偏大如实打印出来。
    print("\n[判据 2] NIS 一致性（3 维观测，理想均值 3）")
    nis_v = nis[m]
    stat = m & (t - t0 < STATIC_SEC)
    mot = m & (t - t0 >= STATIC_SEC)
    nis_stat = float(nis[stat].mean())
    nis_mot = float(nis[mot].mean())
    nis_mean = float(nis_v.mean())
    print(f"  静止段均值 = {nis_stat:.3f}（断言落在 [{NIS_LO}, {NIS_HI}]）")
    print(f"  运动段均值 = {nis_mot:.3f}（防错上界 {NIS_MOTION_MAX}；>3 属 Q 标定问题）")
    print(f"  全程均值 = {nis_mean:.3f}，中位数 = {float(np.median(nis_v)):.3f}，"
          f"95% 分位 = {float(np.percentile(nis_v, 95)):.3f}（χ²(3) 95% 分位 = {CHI2_3_95}）")
    frac_out = float((nis_v > CHI2_3_999).mean())
    print(f"  超过 χ²(3) 99.9% 分位（{CHI2_3_999}）的比例 = {frac_out * 100:.2f}%")
    if not (NIS_LO <= nis_stat <= NIS_HI):
        ok = False
        print(f"  ❌ 静止段均值落在 [{NIS_LO}, {NIS_HI}] 之外。排查方向：")
        print("    - 均值 >> 3：R 偏小或协方差列取错（cov_33/44/55 摊平后下标是 21/28/35，")
        print("      不是 33/44/55 —— 后者超出 36 个元素范围）")
        print("    - 均值 << 3：R 偏大；或姿态插值退化成最近邻")
        print("    - R 对应 Euler 方差而误差状态是旋转向量，小角度下近似等价（ADR-0002）")
    else:
        print(f"  ✅ 静止段均值落在 [{NIS_LO}, {NIS_HI}]：模型假设成立时滤波器是自洽的")
    if nis_mot > NIS_MOTION_MAX:
        ok = False
        print(f"  ❌ 运动段均值超过防错上界 {NIS_MOTION_MAX}：不像单纯的 Q 标定问题，"
              "先查残差是否有尖峰、R 是否取错")
    else:
        print(f"  ⚠️ 运动段均值 {nis_mot:.1f} 明显大于 3：Q 偏小（σ_g 取自静止段），"
              "留待 M5 的 Q 扫描；M4 只需静止段自洽 + 无尖峰")

    # ---- 判据 3：零偏不发散 ----
    # plan.md：初值取静止段估计，全程应基本保持、不发散。判据取两条非任意的界：
    #   ① 终值回到初值 1e-3 以内（plan 说"量级 1e-4"）
    #   ② 全程最大偏离不超过零偏本身的量级（真发散会一直涨下去）
    # 实测运动段有一个暂态偏移（绕圈时被残差推着走，最大 8.4e-3），运动结束后收回 —— 会打印出来。
    print("\n[判据 3] 零偏不发散（初值取静止段估计）")
    dbg = np.abs(bg - bg[0]).max(axis=0)
    bg_scale = float(np.abs(bg[0]).max())
    dbg_all = float(np.abs(bg - bg[0]).max())
    dbg_final = float(np.abs(bg[-1] - bg[0]).max())
    print(f"  初值 b_g = ({bg[0][0]:.6f}, {bg[0][1]:.6f}, {bg[0][2]:.6f}) rad/s")
    print(f"  末值 b_g = ({bg[-1][0]:.6f}, {bg[-1][1]:.6f}, {bg[-1][2]:.6f}) rad/s")
    for k, name in enumerate(("x", "y", "z")):
        print(f"  最大偏离 {name}：{dbg[k]:.3e} rad/s")
    print(f"  终值偏离初值 {dbg_final:.3e}（界 {BG_RECOVER:g}）；"
          f"全程最大偏离 {dbg_all:.3e} / |b_g| {bg_scale:.3e} = {dbg_all / bg_scale:.2f}（界 {BG_BOUNDED:g}）")
    if dbg_final >= BG_RECOVER:
        ok = False
        print(f"  ❌ 终值没回到初值附近：零偏可能真的在发散。排查方向：")
        print("    - 零偏随机游走项 σ_bg²Δt 是否漏加")
        print("    - 残差符号是否反了（方向反了零偏会被持续推向一边）")
    elif dbg_all / bg_scale >= BG_BOUNDED:
        ok = False
        print("  ❌ 瞬时偏离达到零偏本身量级：检查 R 是否取错列、残差是否有尖峰")
    else:
        print(f"  ✅ 不发散：终值回到 {dbg_final:.1e}（plan.md 预期量级 1e-4）")
        if dbg_all > 10 * BG_RECOVER:
            print(f"  ⚠️ 运动段有暂态偏移（最大 {dbg_all:.1e} rad/s，运动结束后收回）："
                  "滤波器把随角速率变化的残差吸收成了零偏，与判据 2 同源（Q 偏小），M5 一并处理")

    # ---- 判据 4：姿态无阶跃 ----
    print("\n[判据 4] 姿态平滑（观测间隙应平滑外推，不被观测打飞）")
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    dots = np.clip(np.abs((qn[:-1] * qn[1:]).sum(axis=1)), 0.0, 1.0)
    step_deg = np.degrees(2 * np.arccos(dots))
    smax = float(step_deg.max())
    istep = int(step_deg.argmax())
    print(f"  相邻帧转角最大值 = {smax:.4f}°（第 {istep} 帧，t = {t[istep + 1] - t0:.2f} s）")
    print(f"  数据真实上限（pose_cov 相邻帧最大转角）= 0.7583°")
    if smax > STEP_TOL_DEG:
        ok = False
        print(f"  ❌ 超过 {STEP_TOL_DEG}°。排查方向：")
        print("    - 观测被错误地当成可信（R 取错列，例如把 cov_33 当成位置协方差）")
        print("    - 残差符号反了导致修正推错方向（翻转帧附近最容易看出来）")
    else:
        print(f"  ✅ 低于 {STEP_TOL_DEG}°")

    # ---------------- 出图 ----------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k, name in enumerate(("z_x", "z_y", "z_z")):
        axes[0].plot(t - t0, z[:, k], lw=0.6, label=name)
    axes[0].set_ylabel("残差 (rad)")
    axes[0].set_title("姿态残差 z = 2·vec(q_hat^-1 ⊗ q_obs) 三分量")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t - t0, z_norm, lw=0.6, color="k")
    if flip_lo is not None:
        for ax in axes:
            ax.axvline(flip_lo - t0, color="r", ls="--", lw=1.0)
        axes[1].annotate("pose 帧 17957 的 q→−q 翻转", xy=(flip_lo - t0, z_norm.max()),
                         xytext=(flip_lo - t0 - 55, z_norm.max() * 0.95),
                         arrowprops=dict(arrowstyle="->", color="r"), color="r", fontsize=9)
    axes[1].set_ylabel("|z| (rad)")
    axes[1].set_xlabel("t (s)")
    axes[1].set_title("残差模长（红线：符号翻转处，应无尖峰）")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "08_eskf_update_residual.png"), dpi=150)
    plt.close(fig)
    print(f"\n图已保存: {fig_dir}/08_eskf_update_residual.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t - t0, nis, lw=0.5, color="C0", alpha=0.7)
    axes[0].axhline(3.0, color="k", ls="-", lw=1.0, label="理想均值 3")
    axes[0].axhline(CHI2_3_95, color="r", ls="--", lw=1.0, label=f"χ²(3) 95% = {CHI2_3_95}")
    axes[0].axhline(nis_mean, color="g", ls=":", lw=1.2, label=f"实测均值 {nis_mean:.2f}")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("NIS = z^T S^-1 z")
    axes[0].set_title("NIS 时序")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(nis_v, bins=80, range=(0, 20), density=True, alpha=0.7, label="实测")
    xs = np.linspace(0.01, 20, 400)
    axes[1].plot(xs, chi2_3_pdf(xs), "r-", lw=1.5, label="χ²(3) 理论")
    axes[1].axvline(nis_mean, color="g", ls=":", lw=1.2, label=f"实测均值 {nis_mean:.2f}")
    axes[1].set_xlabel("NIS")
    axes[1].set_ylabel("概率密度")
    axes[1].set_title("NIS 直方图 vs 卡方分布")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "09_eskf_update_nis.png"), dpi=150)
    plt.close(fig)
    print(f"图已保存: {fig_dir}/09_eskf_update_nis.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    labels = ["P_θx", "P_θy", "P_θz", "P_bgx", "P_bgy", "P_bgz"]
    m_upd = upd == 1
    for k in range(3):
        axes[0].plot(t - t0, P[:, k], lw=0.6, label=labels[k])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("方差（log）")
    axes[0].set_title("姿态误差方差 P_θθ：观测每帧把它压回下限（M3 无观测时只增不减）")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)
    # 放大看锯齿：R 很小 + 200 Hz 更新把 P 钉在下限附近，全程图上看不出"预测涨、观测降"，
    # 必须放大到 ~0.3 s 才看得见（ref/02 说这是 ESKF 最有辨识度的图）。
    ax_in = axes[0].inset_axes([0.62, 0.12, 0.36, 0.36])
    w = np.where(m_upd)[0]
    i0 = int(w[len(w) // 2]) if len(w) else 0
    sel = slice(i0, min(i0 + 60, len(t)))
    ax_in.plot(t[sel] - t0, P[sel, 0], ".-", ms=2, lw=0.7, label="P_θx")
    ax_in.plot(t[sel] - t0, P[sel, 1], ".-", ms=2, lw=0.7, label="P_θy")
    ax_in.set_title("放大 0.3 s：预测涨 / 观测降", fontsize=7)
    ax_in.tick_params(labelsize=6)
    ax_in.legend(fontsize=6, ncol=2)
    axes[0].indicate_inset_zoom(ax_in, edgecolor="gray")
    for k, name in enumerate(("bg_x", "bg_y", "bg_z")):
        axes[1].plot(t - t0, bg[:, k], lw=0.8, label=name)
    axes[1].set_ylabel("b_g (rad/s)")
    axes[1].set_xlabel("t (s)")
    axes[1].set_title("陀螺零偏估计：静止段（0–60 s）基本保持；运动段被残差推动（Q 偏小，M5 处理），"
                      "110 s 后收回")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "10_eskf_update_P_bg.png"), dpi=150)
    plt.close(fig)
    print(f"图已保存: {fig_dir}/10_eskf_update_P_bg.png")

    if not ok:
        sys.exit(1)
    print("\nM4 四项判据全部通过 ✅")


if __name__ == "__main__":
    main()
