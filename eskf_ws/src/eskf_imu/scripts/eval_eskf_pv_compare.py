#!/usr/bin/env python3
"""选做（M7/M8）的两组对照实验 —— "一组实验结果"里最能说明问题的两张图。

对照 A：**位置观测开关**（姿态观测两组都开，只差位置观测）
    只预测不修正（纯加计双重积分）→ 128 s 内位置飘到 800 m 量级
    接上位置观测            → 全程贴着观测（误差 mm 级）
    结论：位置估计之所以能工作，靠的是观测；滤波器自己的贡献是速度、加计零偏、
    以及观测间隙的外推。

对照 B：**加计零偏初值**（与 M5 的 b_g 初值对照同构）
    初值取静止段估计 → b_a 曲线平坦
    初值取 0         → b_a 从 0 爬到同一个值（末值差 < 1e-3 m/s²）
    结论：位置观测确实在估计 b_a，而不是让初值原样保持。

用法（不依赖 ROS）：
    python3 scripts/eval_eskf_pv_compare.py output/ekf_pv_out.csv output/ekf_pv_noobs.csv \
        output/ekf_pv_ba0.csv ../../../data/pose_cov.csv output

出图 20_pv_ablation.png。
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

TRUE_MAX_NORM = 8.94    # 实测轨迹 |p| 最大（plan.md §1.4：绕行约 9 m）
DRIFT_MIN = 100.0       # 无位置观测组的末 |p| 至少应飘到多少（m）
MAIN_END_MAX = 0.1      # 主结果末值 |p| 的上限（m，数据本身绕一圈回原点）
MAIN_MAX_TOL = 0.2      # 主结果 |p| 最大值与真实轨迹的允许相对偏差
BA_AGREE = 1e-3         # 两组 b_a 末值之差上限（m/s²）
POS_AGREE = 5e-3        # 两组位置末值之差上限（m）


def read_csv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        rows = np.array(
            [[float(v) for v in line.strip().split(",")] for line in f if line.strip()]
        )
    return rows, idx


def take(d, di, cols, t0):
    t = d[:, di["time"]] - t0
    return t, np.stack([d[:, di[c]] for c in cols], axis=1)


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main_path, noobs_path, ba0_path = sys.argv[1], sys.argv[2], sys.argv[3]
    pose_path = sys.argv[4]
    out_dir = sys.argv[5] if len(sys.argv) > 5 else "output"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    m, mi = read_csv(main_path)
    n, ni = read_csv(noobs_path)
    b, bi = read_csv(ba0_path)
    ob, oi = read_csv(pose_path)
    for name, idx in (("主结果", mi), ("无位置观测", ni), ("b_a 初值 0", bi)):
        if "ba_x" not in idx:
            sys.exit(f"{name}: 缺 ba_x 列 —— 这三份 CSV 都要开 estimate_position:=true")
    t0 = m[0, mi["time"]]
    t_m, p_m = take(m, mi, ("x", "y", "z"), t0)
    t_n, p_n = take(n, ni, ("x", "y", "z"), t0)
    t_b, p_b = take(b, bi, ("x", "y", "z"), t0)
    _, ba_m = take(m, mi, ("ba_x", "ba_y", "ba_z"), t0)
    _, ba_b = take(b, bi, ("ba_x", "ba_y", "ba_z"), t0)
    t_o = ob[:, oi["time"]] - t0
    p_o = np.stack([ob[:, oi[c]] for c in ("x", "y", "z")], axis=1)

    n_m = np.linalg.norm(p_m, axis=1)
    n_n = np.linalg.norm(p_n, axis=1)
    n_o = np.linalg.norm(p_o, axis=1)
    print(f"轨迹合模长：观测最大 {n_o.max():.3f} m（真实轨迹约 {TRUE_MAX_NORM} m）")
    print(f"  主结果（接位置观测）：最大 {n_m.max():.3f} m，末值 {n_m[-1] * 1e3:.2f} mm")
    print(f"  只预测（不接位置观测）：{', '.join(f'{s} s 时 {n_n[int(np.searchsorted(t_n, s))]:.1f} m' for s in (20, 40, 60, 80, 100, 127))}")
    print(f"     末值 {n_n[-1]:.1f} m，末速度 |v| = "
          f"{np.linalg.norm(np.stack([n[:, ni[c]] for c in ('vx', 'vy', 'vz')], axis=1)[-1]):.2f} m/s")
    print("     末帧 P_p 对角线 = "
          + ", ".join(f"{n[-1, ni[c]]:.3e}" for c in ("Ppx", "Ppy", "Ppz"))
          + " m²（滤波器自己也清楚自己有多不确定）")
    print("     主结果末帧 P_p = "
          + ", ".join(f"{m[-1, mi[c]]:.3e}" for c in ("Ppx", "Ppy", "Ppz")) + " m²")

    ok = True
    print("\n[判据 1] 位置观测是位置估计能工作的原因")
    if n_n[-1] > DRIFT_MIN and n_m[-1] < MAIN_END_MAX and \
            abs(n_m.max() - TRUE_MAX_NORM) < MAIN_MAX_TOL * TRUE_MAX_NORM:
        print(f"  ✅ 关掉位置观测后末值飘到 {n_n[-1]:.0f} m（真实轨迹只有 {TRUE_MAX_NORM} m）；"
              f"接上后最大 {n_m.max():.3f} m、末值 {n_m[-1] * 1e3:.1f} mm —— 全程贴着真实轨迹")
    else:
        ok = False
        print(f"  ❌ 预期：无位置观测组末值 > {DRIFT_MIN} m、主结果末值 < {MAIN_END_MAX} m 且"
              f"最大 |p| ≈ {TRUE_MAX_NORM} m。实际 {n_n[-1]:.1f} / {n_m[-1]:.3f} / "
              f"{n_m.max():.3f}。排查方向：")
        print("    - use_position_update:=false 那组是否真的没做位置更新（看日志的『位置观测更新』帧数）")
        print("    - 主结果是否漏了位置观测（upd_p 列应几乎全 1）")

    print("\n[判据 2] b_a 初值取 0 也能收敛到同一个值（观测真的在估计它）")
    d_ba = np.abs(ba_m[-1] - ba_b[-1])
    print(f"  两组 b_a 末值：{' / '.join(f'{v:+.6f}' for v in ba_m[-1])}"
          f"  vs  {' / '.join(f'{v:+.6f}' for v in ba_b[-1])}")
    print(f"  末值差 = ({d_ba[0]:.2e}, {d_ba[1]:.2e}, {d_ba[2]:.2e}) m/s²（界 {BA_AGREE:g}）")
    for k, nm in enumerate("xyz"):
        print(f"  b_a_{nm}：{ba_b[0, k]:+.6f} → 1 s {ba_b[int(np.searchsorted(t_b, 1.0)), k]:+.6f}"
              f" → 10 s {ba_b[int(np.searchsorted(t_b, 10.0)), k]:+.6f}"
              f" → 末 {ba_b[-1, k]:+.6f}")
    if d_ba.max() < BA_AGREE:
        print("  ✅ 一致：初值只影响收敛过程，不影响终点")
    else:
        ok = False
        print("  ❌ 末值不一致：σ_ba（加计零偏随机游走）是否漏了？或该组没在估计 b_a")

    print("\n[判据 3] 两组位置结果一致（b_a 初值不同不应改变位置）")
    dp = np.abs(p_m[-1] - p_b[-1]).max()
    dp_all = np.abs(p_m - p_b).max()
    print(f"  末值差最大 {dp * 1e3:.4f} mm；全程最大差 {dp_all * 1e3:.3f} mm（界 {POS_AGREE * 1e3:.0f} mm）")
    if dp_all < POS_AGREE:
        print("  ✅ 位置几乎不受 b_a 初值影响（因为位置被观测钉着，差的只是 b_a 的收敛过程）")
    else:
        ok = False
        print("  ❌ 位置差异过大：检查两组是否只有 acc_bias_init 不同")

    # ---------------- 出图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.semilogy(t_o, n_o, lw=0.8, color="C2", ls="--", label="FAST-LIO 观测")
    ax.semilogy(t_m, n_m, lw=0.9, color="C0", label="ESKF（接位置观测）")
    ax.semilogy(t_n, n_n, lw=0.9, color="C3", label="只预测（不接位置观测）")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|p| (m, log)")
    ax.set_title(f"位置观测开关对照：关掉后飘到 {n_n[-1]:.0f} m\n"
                 f"（真实轨迹只有 {TRUE_MAX_NORM} m）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(p_o[:, 0], p_o[:, 1], lw=0.8, color="C2", ls="--", label="观测轨迹")
    ax.plot(p_m[:, 0], p_m[:, 1], lw=0.9, color="C0", label="ESKF（接位置观测）")
    ax.plot(p_n[:, 0], p_n[:, 1], lw=0.7, color="C3", label="只预测（不接）")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("xy 轨迹：没观测的那条已经跑到 800 m 之外", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for k, nm in enumerate(("x", "y", "z")):
        ax.plot(t_m, ba_m[:, k], lw=1.0, color=f"C{k}", label=f"初值=静止段估计: b_a_{nm}")
        ax.plot(t_b, ba_b[:, k], lw=0.8, ls=":", color=f"C{k}", alpha=0.9,
                label=f"初值=0: b_a_{nm}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("b_a (m/s²)")
    ax.set_title(f"b_a 初值对照：两组末值差 < {d_ba.max():.1e} m/s²", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "20_pv_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"\n图已存: {fig_dir}/20_pv_ablation.png")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
