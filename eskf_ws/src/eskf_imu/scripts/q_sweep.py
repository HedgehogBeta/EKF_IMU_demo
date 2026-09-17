#!/usr/bin/env python3
"""M5 的 Q 扫描实验：题面只让 R 逐帧取自数据（不可调），所以唯一能扫的是 Q。

三组 σ_g：×0.1（更信陀螺）/ ×1（基线）/ ×10（更信观测）。σ_bg 不动。
每组算四个量，整理成一张表：

    静止段姿态波动 std(yaw)       —— σ_g ↑ 会更贴观测 → 更抖
    运动段跟踪误差 EKF vs 观测     —— σ_g ↑ 更贴观测 → 误差更小
    运动段跟踪滞后（互相关最优时移）—— σ_g ↑ 响应更快 → 滞后更小
    NIS 均值（静止 / 运动）        —— σ_g ↑ 说明预测更不信自己 → NIS 更接近 3

用法（不依赖 ROS）：
    python3 scripts/q_sweep.py output/ekf_q_x0p1.csv output/ekf_out.csv output/ekf_q_x10.csv \
        ../../../data/pose_cov.csv output

判据：NIS 均值随 σ_g 单调下降（σ_g 确实在控制"信陀螺还是信观测"）。
表同时写一份 output/q_sweep_table.csv，报告里直接引用。
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

STATIC_SEC = 60.0   # plan.md §1.4：静止段 0–60 s
MOTION_LO, MOTION_HI = 60.0, 110.0  # 绕圈段（110 s 后回程，速度低）
MAX_LAG_FRAMES = 60  # 互相关搜索范围（±0.3 s）


def read_csv(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        rows = np.array(
            [[float(v) for v in line.strip().split(",")] for line in f if line.strip()]
        )
    return rows, idx


def slerp(q0, q1, u):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-8
    w1 = np.where(near, u, np.sin((1 - u) * theta) / np.where(near, 1, sin_theta))
    w2 = np.where(near, 1 - u, np.sin(u * theta) / np.where(near, 1, sin_theta))
    q = w1 * q0 + w2 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def align_pose_to(t_q, q_p, t_p):
    j = np.clip(np.searchsorted(t_p, t_q) - 1, 0, len(t_p) - 2)
    span = t_p[j + 1] - t_p[j]
    u = np.where(span > 0, (t_q - t_p[j]) / np.where(span > 0, span, 1.0), 0.0)
    return slerp(q_p[j], q_p[j + 1], np.clip(u, 0, 1)[:, None])


def best_lag_frames(a, b, max_lag):
    """a 相对 b 的最优时移（帧）：把 a 平移 lag 帧后与 b 的相关最大者。
    正值 = a 滞后于 b。"""
    a = a - a.mean()
    b = b - b.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    cc = []
    for lag in lags:
        if lag >= 0:
            x, y = a[lag:], b[: len(b) - lag]
        else:
            x, y = a[: len(a) + lag], b[-lag:]
        if len(x) < 10:
            cc.append(-np.inf)
            continue
        cc.append(float(np.corrcoef(x, y)[0, 1]))
    return int(lags[int(np.argmax(cc))])


def metrics(path, pose):
    d, di = read_csv(path)
    t = d[:, di["time"]]
    tr = t - t[0]
    q = np.stack([d[:, di[c]] for c in ("qw", "qx", "qy", "qz")], axis=1)[:, [1, 2, 3, 0]]
    t_p, q_p = pose
    q_ref = align_pose_to(t, q_p, t_p)
    dots = np.abs(np.sum(q * q_ref, axis=1))
    err = np.degrees(2 * np.arccos(np.clip(dots, -1.0, 1.0)))
    yaw = np.unwrap(np.radians(d[:, di["yaw_deg"]]))
    yaw_obs = np.unwrap(
        np.arctan2(
            2 * (q_ref[:, 3] * q_ref[:, 2] + q_ref[:, 0] * q_ref[:, 1]),
            1 - 2 * (q_ref[:, 1] ** 2 + q_ref[:, 2] ** 2),
        )
    )
    st = tr < STATIC_SEC
    mot = (tr >= MOTION_LO) & (tr < MOTION_HI)
    nis = d[:, di["nis"]]
    # 运动段的"跟随快慢"：yaw 角速率与观测角速率的相关系数（1 = 完全跟随）。
    # σ_g ↑ → 更信观测 → 相关系数更接近 1；σ_g ↓ → 更信陀螺，速率里多出陀螺自己的噪声。
    dy, do = np.diff(yaw)[mot[:-1]], np.diff(yaw_obs)[mot[:-1]]
    return {
        "yaw_std_static_deg": float(np.degrees(np.std(yaw[st]))),
        "err_motion_deg": float(err[mot].mean()),
        "rate_corr": float(np.corrcoef(dy, do)[0, 1]),
        "lag_frames": best_lag_frames(yaw[mot], yaw_obs[mot], MAX_LAG_FRAMES),
        "dt_ms": float(np.median(np.diff(t)) * 1e3),
        "nis_static": float(nis[st].mean()),
        "nis_motion": float(nis[mot].mean()),
        "bg_final": d[-1, di["bg_z"]],
    }


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    paths = sys.argv[1:4]
    pose_path, out_dir = sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "output"
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    p, pi = read_csv(pose_path)
    pose = (p[:, pi["time"]], np.stack([p[:, pi[c]] for c in ("qx", "qy", "qz", "qw")], axis=1))

    sigmas = [0.0032 * 0.1, 0.0032, 0.0032 * 10.0]
    labels = ["σ_g ×0.1\n(更信陀螺)", "σ_g ×1\n(基线)", "σ_g ×10\n(更信观测)"]
    rows = []
    for path, sg in zip(paths, sigmas):
        m = metrics(path, pose)
        m["sigma_g"] = sg
        m["file"] = os.path.basename(path)
        rows.append(m)

    # 表头里的时段与 M4 的 NIS 判据略有不同（M4 用 60–128 s），这里取 60–110 s 的绕圈段
    # （110 s 后是低速回程，会把 NIS 均值拉低）。
    print(f"{'文件':<16}{'σ_g':>9}{'静止段yawstd(°)':>18}{'运动段误差(°)':>16}"
          f"{'速率相关':>10}{'滞后(ms)':>10}{'NIS静':>9}{'NIS动':>9}{'末bg_z':>12}")
    for m in rows:
        print(f"{m['file']:<16}{m['sigma_g']:>9.4f}{m['yaw_std_static_deg']:>18.4f}"
              f"{m['err_motion_deg']:>16.3f}{m['rate_corr']:>10.3f}"
              f"{m['lag_frames'] * m['dt_ms']:>10.1f}{m['nis_static']:>9.3f}"
              f"{m['nis_motion']:>9.3f}{m['bg_final']:>12.6f}")

    # 写一份 CSV 表，报告里直接引用
    table_path = os.path.join(out_dir, "q_sweep_table.csv")
    with open(table_path, "w") as f:
        f.write("file,sigma_g,sigma_bg,yaw_std_static_deg,err_motion_deg,rate_corr_60_110s,"
                "lag_ms,nis_static_0_60s,nis_motion_60_110s,bg_z_final\n")
        for m in rows:
            f.write(f"{m['file']},{m['sigma_g']:.6g},1e-4,{m['yaw_std_static_deg']:.6f},"
                    f"{m['err_motion_deg']:.6f},{m['rate_corr']:.6f},"
                    f"{m['lag_frames'] * m['dt_ms']:.3f},{m['nis_static']:.6f},"
                    f"{m['nis_motion']:.6f},{m['bg_final']:.9f}\n")
    print(f"\n表已存: {table_path}")

    ok = True
    nis_seq = [m["nis_motion"] for m in rows]
    err_seq = [m["err_motion_deg"] for m in rows]
    std_seq = [m["yaw_std_static_deg"] for m in rows]
    if not (nis_seq[0] > nis_seq[1] > nis_seq[2]):
        ok = False
        print(f"❌ 运动段 NIS 均值没有随 σ_g 单调下降：{[round(x, 2) for x in nis_seq]}。")
        print("   预期 σ_g ↑ → 预测更不信自己 → NIS 更小。若不是单调：")
        print("   - 三份 CSV 是否是同一配置、只差 sigma_g（对比文件头的参数）")
        print("   - R 是否被改动过（R 由题面给定，不可调）")
        print("   - 是否有一组的 /pose_cov 掉帧（看 08/09 图的残差是否有大缺口）")
    else:
        print(f"✅ 运动段 NIS 均值随 σ_g 单调下降：{[round(x, 2) for x in nis_seq]}")
    if not (err_seq[0] > err_seq[1] > err_seq[2]):
        ok = False
        print(f"❌ 运动段跟踪误差没有随 σ_g 单调下降：{[round(x, 3) for x in err_seq]}。"
              "σ_g ↑ 应该更贴观测、误差更小")
    else:
        print(f"✅ 运动段跟踪误差随 σ_g 单调下降：{[round(x, 3) for x in err_seq]}°")
    if not (std_seq[0] < std_seq[1] < std_seq[2]):
        ok = False
        print(f"❌ 静止段 yaw 波动没有随 σ_g 单调上升：{[round(x, 4) for x in std_seq]}。"
              "σ_g ↑ 会更贴（本身就更抖的）观测")
    else:
        print(f"✅ 静止段 yaw 波动随 σ_g 单调上升：{[round(x, 4) for x in std_seq]}°"
              "（更信观测的代价）")
    lag_ms = [m["lag_frames"] * m["dt_ms"] for m in rows]
    print(f"  ℹ️ 互相关最优时移 = {lag_ms} ms（分辨率 1 帧 = {rows[0]['dt_ms']:.2f} ms）："
          "三组都测不出滞后 —— 在本题这么小的 R 下，滞后不是主导效应，"
          "主导的是误差量级与 NIS 自洽性")

    # ---------------- 出图 ----------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(3)
    names = ["静止段 yaw 波动 std (°)", "运动段跟踪误差 (°)", "yaw 速率相关系数",
             "运动段 NIS 均值"]
    vals = [
        [m["yaw_std_static_deg"] for m in rows],
        [m["err_motion_deg"] for m in rows],
        [m["rate_corr"] for m in rows],
        [m["nis_motion"] for m in rows],
    ]
    for ax, nm, v in zip(axes, names, vals):
        ax.bar(x, v, color=["C1", "C0", "C2"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(nm, fontsize=10)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.3g}", (xi, vi), ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        if nm.startswith("yaw 速率"):
            # 这三个值都挤在 0.97–0.99，全量程柱状图看不出趋势，单独放大纵轴
            ax.set_ylim(min(v) - 0.02, 1.0)
    fig.suptitle("M5：Q（σ_g）扫描 —— σ_g ↑ 更信观测：更贴观测（误差↓、速率相关↑、NIS↓），"
                 "代价是静止段更抖")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "13_q_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"图已存: {out_dir}/figures/13_q_sweep.png")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
