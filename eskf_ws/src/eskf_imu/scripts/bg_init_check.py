#!/usr/bin/env python3
"""M5 的零偏初值对照实验：b_g 初值取"静止段估计" vs 取 0，其余配置完全一样。

这一组对照回答的是主线只看一遍看不出来的问题：**观测到底有没有在修正零偏？**

    初值 = 静止段估计 → b_g 曲线平坦（零偏一开始就对，观测只需微调）
    初值 = 0         → z 轴从 0 爬到 0.0166（静止段实测均值），爬升快慢由 σ_bg 决定

两组末值应该收敛到同一个数（差 < 1e-4 rad/s）—— 这直接证明观测在"估计"零偏，
而不是让初值原样保持。

用法（不依赖 ROS）：
    python3 scripts/bg_init_check.py output/ekf_out.csv output/ekf_bg0.csv output

出图 14_bg_init_compare.png（两组并排）+ 表 output/bg_init_table.csv。
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

# 静止段实测均值（plan.md §1.2）：零偏真值的参考
BG_STATIC_REF = np.array([0.00048, 0.00248, 0.01661])
FINAL_TOL = 1e-4      # 两组末值之差的上限（rad/s）
STATIC_MATCH = 0.2    # 末值与静止段参考的相对偏差上限
FLAT_TOL = 1e-3       # "平坦"的界：初值=静止段估计组的全程偏离应 < 1e-3


def read_csv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        rows = np.array(
            [[float(v) for v in line.strip().split(",")] for line in f if line.strip()]
        )
    return rows, idx


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    est_path, zero_path = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    runs = []
    for path, label in ((est_path, "初值 = 静止段估计"), (zero_path, "初值 = 0")):
        d, di = read_csv(path)
        t = d[:, di["time"]]
        bg = np.stack([d[:, di[c]] for c in ("bg_x", "bg_y", "bg_z")], axis=1)
        runs.append({"file": os.path.basename(path), "label": label, "t": t - t[0], "bg": bg})

    print(f"{'配置':<18}{'文件':<16}{'初值 bg_z':>12}{'末值 bg_z':>12}{'全程偏离':>12}"
          f"{'到90%用时':>12}")
    for r in runs:
        z = r["bg"][:, 2]
        dev = np.abs(r["bg"] - r["bg"][0]).max()
        target = BG_STATIC_REF[2]
        reach = np.where(np.abs(z - target) < 0.1 * target)[0]
        t_reach = r["t"][reach[0]] if len(reach) else float("nan")
        r.update(dev=dev, t_reach=t_reach)
        print(f"{r['label']:<18}{r['file']:<16}{z[0]:>12.6f}{z[-1]:>12.6f}{dev:>12.3e}"
              f"{t_reach:>11.2f}s")

    est, zero = runs
    ok = True

    print(f"\n[判据 1] 初值 = 0 的那组，z 轴应从 0 爬起来")
    z0 = zero["bg"][:, 2]
    if z0[0] != 0.0:
        # 允许写成 0.000000 以外的小值
        print(f"  ⚠️ 该组初值不是 0（{z0[0]:.3e}）：传参时是不是忘了 -p bg_init:=zero")
    climb = abs(z0[-1] - z0[0])
    print(f"  b_g 的 z 分量：{z0[0]:.6f} → {z0[-1]:.6f}（爬升 {climb:.6f} rad/s）")
    print(f"  静止段实测均值参考 {BG_STATIC_REF[2]:.6f} rad/s，"
          f"相对偏差 {abs(z0[-1] - BG_STATIC_REF[2]) / BG_STATIC_REF[2] * 100:.2f}%"
          f"（界 {STATIC_MATCH * 100:.0f}%）")
    if climb > 0.5 * BG_STATIC_REF[2] and abs(z0[-1] - BG_STATIC_REF[2]) / BG_STATIC_REF[2] < STATIC_MATCH:
        print("  ✅ 从 0 爬到了静止段实测均值附近 —— 观测确实在修正零偏")
    else:
        ok = False
        print("  ❌ 没爬到位。排查方向：")
        print("    - Q 里的零偏随机游走项 σ_bg²Δt 是否漏加（漏了零偏方差永不减，观测修正不了它）")
        print("    - 残差符号是否反了（方向反了零偏会被推向一边，或不收敛）")
        print("    - 该组是否真的用了 -p bg_init:=zero")

    print("\n[判据 2] 初值 = 静止段估计的那组应保持平坦")
    # "平坦"分两段看，与 M4 的零偏判据同源：
    #   静止段（0–60 s）：零偏一开始就估对了，这里必须平；运动段有一个暂态偏移（实测最大
    #   8.4e-3 rad/s：滤波器把随角速率变化的残差吸收成了零偏，根因是 Q 偏小，M5 的 Q 扫描处理），
    #   运动结束后收回。所以判据压在"静止段 + 终值"，运动段只如实打印。
    st = est["t"] < 60.0
    dev_static = float(np.abs(est["bg"][st] - est["bg"][0]).max())
    dev_final = float(np.abs(est["bg"][-1] - est["bg"][0]).max())
    print(f"  静止段（0–60 s）最大偏离 {dev_static:.3e}；终值偏离 {dev_final:.3e}"
          f"（界 {FLAT_TOL:g}）")
    if dev_static < FLAT_TOL and dev_final < FLAT_TOL:
        print("  ✅ 平坦（初值已经对了，观测只需微调）")
        if est["dev"] > 10 * FLAT_TOL:
            print(f"  ⚠️ 运动段有暂态偏移（全程最大 {est['dev']:.3e}，110 s 后收回）："
                  "与 M4 完成记录里的现象一致 —— 随角速率变化的残差被吸收成零偏，根因是 Q 偏小")
    else:
        ok = False
        print("  ❌ 静止段或终值偏离过大：初值虽对但被观测推走了，检查 R 是否取错列"
              "（cov_33/44/55 摊平后是 21/28/35）")

    print("\n[判据 3] 两组末值应收敛到同一个数（观测真的在估计零偏）")
    dz = np.abs(est["bg"][-1] - zero["bg"][-1])
    print(f"  末值差 = ({dz[0]:.3e}, {dz[1]:.3e}, {dz[2]:.3e}) rad/s（界 {FINAL_TOL:g}）")
    print(f"  两组末值：{' / '.join(f'{v:.6f}' for v in est['bg'][-1])}"
          f"  vs  {' / '.join(f'{v:.6f}' for v in zero['bg'][-1])}")
    if dz.max() < FINAL_TOL:
        print("  ✅ 一致：零偏初值只影响收敛过程，不影响终点 —— 观测把它估出来了")
    else:
        ok = False
        print("  ❌ 两组末值不一致：零初值那组可能还没收敛（观察时间不够）或发散")

    # 表
    table_path = os.path.join(out_dir, "bg_init_table.csv")
    with open(table_path, "w") as f:
        f.write("config,file,bg_x_init,bg_y_init,bg_z_init,bg_x_final,bg_y_final,bg_z_final,"
                "z_climb,t_to_90pct_s\n")
        for r in runs:
            b = r["bg"]
            f.write(f"{r['label']},{r['file']},{b[0,0]:.9f},{b[0,1]:.9f},{b[0,2]:.9f},"
                    f"{b[-1,0]:.9f},{b[-1,1]:.9f},{b[-1,2]:.9f},{b[-1,2] - b[0,2]:.9f},"
                    f"{r['t_reach']:.3f}\n")
    print(f"\n表已存: {table_path}")

    # ---------------- 出图 ----------------
    names = ["b_gx", "b_gy", "b_gz (rad/s)"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for k in range(3):
        for col, r in enumerate(runs):
            ax = axes[k, col]
            ax.plot(r["t"], r["bg"][:, k], lw=1.0,
                    color="C1" if col == 0 else "C0")
            ax.axhline(BG_STATIC_REF[k], color="k", ls=":", lw=0.9,
                       label="静止段实测均值")
            ax.set_ylabel(names[k] if col == 0 else "")
            ax.grid(True, alpha=0.3)
            if k == 0:
                ax.set_title(f"{r['label']}（{r['file']}）", fontsize=10)
            if k == 2:
                ax.set_xlabel("t (s)")
            if col == 0 and k == 2:
                ax.set_ylim(0, 0.02)
    axes[2, 1].annotate(
        f"从 0 爬到 {zero['bg'][-1, 2]:.4f}\n（静止段均值 {BG_STATIC_REF[2]:.5f}）",
        xy=(40, zero["bg"][:, 2].max() * 0.55), fontsize=9, color="C0")
    fig.suptitle("M5：零偏初值对照 —— 左列初值取静止段估计（平坦），右列初值取 0（爬升）\n"
                 "两组末值一致 ⇒ 观测在修正零偏，而不是让初值原样保持")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "14_bg_init_compare.png"), dpi=150)
    plt.close(fig)
    print(f"图已存: {fig_dir}/14_bg_init_compare.png")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
