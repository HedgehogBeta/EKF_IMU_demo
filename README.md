# 基于 EKF 的 IMU 姿态解算

传感器组考核题目二：用 ESKF 从六轴 IMU 数据估计姿态（Roll/Pitch/Yaw），观测来自 FAST-LIO
输出的位姿（`data/pose_cov.csv`）。

误差状态取 15 维 $\delta x = [\delta p, \delta v, \delta\theta, \delta b_g, \delta b_a]$


## 构建运行

```bash
cd eskf_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

一键回放（起 ESKF 节点 + 两个回放节点，回放完自动收尾）：

```bash
cd eskf_ws/src/eskf_imu
bash scripts/run_replay.sh $PWD/output/ekf_pv_out.csv
```

节点日志留在 `<输出CSV>.log`，验收数字全在里面。手跑三个终端也可以，注意
**节点要比 player 早启动几秒**（ROS2 发现需要时间，晚订阅会丢掉开头的帧）：

```bash
ros2 run eskf_imu eskf_node --ros-args -p csv_out:=$PWD/output/ekf_pv_out.csv
ros2 run eskf_imu pose_player_node --ros-args -p csv_path:="$DATA/pose_cov.csv" -p speed:=20.0
ros2 run eskf_imu imu_player_node  --ros-args -p csv_path:="$DATA/imu.csv"     -p speed:=20.0
```

`output/` 是回放产物（CSV、日志、图），不进版本控制，跑一次就有。


## 出图与验收

```bash
cd eskf_ws/src/eskf_imu
# 曲线：Roll/Pitch/Yaw–Time 与 x/y/z–Time
python3 scripts/plot.py output/ekf_pv_out.csv ../../../data/pose_cov.csv output
# 数值验收：姿态 2 条 + 位置 6 条判据，不达标以 1 退出
python3 scripts/eval_eskf_pv.py output/ekf_pv_out.csv ../../../data/pose_cov.csv output
```

| 图 | 内容 |
|---|---|
| `01_rpy_time.png` | Roll/Pitch/Yaw–Time（题面 §4(4) 要求的三条曲线） |
| `02_position_time.png` | x/y/z–Time（选做） |
| `03_attitude_error.png` | 姿态测地角误差时序 / 分布 / 60–70 s 放大 |
| `04_trajectory.png` | xy 平面轨迹 + 位置跟踪误差 |
| `05_velocity_bias.png` | 速度估计与加计零偏 |
| `06_residual_nis.png` | 位置残差与 NIS 时序 |
| `07_covariance.png` | P 的 p/v/b_a 块对角线（预测涨、观测降） |


## 一组实验结果

128.1 s、25614 帧（dt 中位 4.918 ms）。姿态观测更新 25559 帧、位置观测更新 25559 帧，
开头 55 帧无观测（`/pose_cov` 比 IMU 晚 0.17 s，且其前 20 帧协方差全零）。

| 指标 | 数值 |
|---|---|
| 姿态测地角误差 均值 / 最大 | **0.037°** / 0.456°（验收线 5°） |
| 姿态平滑度（yaw 二阶差分 std） | 3.81e-4 rad，比它跟随的观测平滑 39.6% |
| 位置跟踪 RMS（x/y/z） | **1.01 / 1.35 / 0.76 mm**（观测自报噪声 √cov 中位 ≈1.1–1.3 mm） |
| 位置跟踪最大误差 | 9.34 mm |
| 位置 NIS 静止段 / 运动段 | 1.077 / 5.746（理论均值 3） |
| 速度 \|v\|max | 0.970 m/s（观测平滑差分参考 0.963，比值 1.007） |
| 加计零偏 $b_a$ 末值 | (−0.0395, +0.0049, −0.0573) m/s²（全程有界，\|b_a\|max 0.174 m/s²） |
| 终点偏差 \|p̂_end\| | 25.2 mm（轨迹本身绕一圈回到原点） |

$b_g$ 末值 (0.000348, 0.002406, 0.016762) rad/s，与静止段估计同量级。

姿态误差是"估计与观测之差"，位置输出按构造跟随位置观测，所以位置那几行是**自洽性**检验
（跟踪误差落在观测自身噪声量级、NIS 在合理区间、不发散），不是绝对精度。速度与 $b_a$
不是被观测的量，只有它们才是滤波器自己算出来的东西。


## EKF 说明

**状态量。** 标称状态 $p, v, q, b_g, b_a$，即位置、速度、姿态四元数、陀螺零偏、加计零偏。
滤波器估计 15 维误差状态 $\delta x = [\delta p, \delta v, \delta\theta, \delta b_g, \delta b_a]$，
姿态误差用旋转向量，$P$ 为 15×15。。

**预测模型。** 每来一帧 IMU，先用零偏补偿后的陀螺积分姿态，再用去零偏、转到世界系、
减去重力的加计积分速度，速度再积分位置：

$$\omega = \omega_m - b_g,\qquad q \leftarrow q \otimes \mathrm{Exp}(\omega\Delta t)$$
$$\tilde a = a_m - b_a,\qquad a_W = R\tilde a - g_W,\qquad v \leftarrow v + a_W\Delta t,\qquad p \leftarrow p + v\Delta t$$

四元数更新后归一化。加速度原始单位是 g，节点内统一换成 m/s²。协方差按 $P \leftarrow FPF^\top + Q$ 传播，$F$ 由 $I + A_c\Delta t$ 得到，
只保留三处主要耦合：位置误差随速度误差累积、速度误差由姿态误差和加计零偏引起、姿态
误差由陀螺零偏引起。零偏本身不主动变，其不确定度只靠 $Q$ 增长。

**观测模型。** 观测是 FAST-LIO 输出的位姿：姿态取四元数 $q_x,q_y,q_z,q_w$，位置取
$x,y,z$。每个观测只与对应的那 3 个误差分量有关，$H$ 里其余元素全为 0。位置残差就是
向量差；姿态残差用旋转向量差而不是四元数分量差，顺便避开 $q$ 与 $-q$ 表示同一旋转带来
的符号问题。两者的 $R$ 都逐帧取自 CSV 的协方差对角线：姿态 `cov_33/44/55`、位置
`cov_00/11/22`（**这是列名**，摊平成 6×6 后下标是 21/28/35 与 0/7/14）。更新为标准的
$S = HPH^\top + R$、$K = PH^\top S^{-1}$、$\delta x = Kz$，注入后清零，$P \leftarrow (I-KH)P$。

**Q 与 R 的作用。** 两者决定滤波器在「信模型」与「信观测」之间怎么权衡。$R$ 来自观测自身
的不确定度、逐帧变化、不可调：取小了观测噪声会灌进状态（姿态发毛、位置阶跃），取大了
滤波器跟不上真值、退化成自由积分。$Q$ 是模型噪声，含陀螺白噪声、加计白噪声和两项零偏
随机游走（题面要求），前两项按 $\Delta t^2$ 计入、零偏项按 $\Delta t$ 计入。$Q$ 偏小则
滤波器过度自信、对观测不敏感（残差持续偏大、NIS 冲高）；偏大则噪声灌进状态。



## 节点参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `csv_out` | `output/ekf_out.csv` | 输出 CSV 路径 |
| `bias_window_frames` | 1000 | 静止段初始化窗口（前 N 帧均值定 $b_g$、初始 roll/pitch、$b_a$ 初值） |
| `g_ms2` | 9.80665 | 加计读数 g → m/s² 的换算系数，同时是重力向量 $(0,0,+\lvert g\rvert)$ 的模长 |
| `sigma_g` / `sigma_bg` | 0.0032 / 1e-4 | 陀螺白噪声 std（rad/s）、零偏随机游走（rad/s/√s） |
| `sigma_a` / `sigma_ba` | 0.12 / 5e-4 | 加计白噪声 std（m/s²）、零偏随机游走（m/s²/√s） |
| `p0_theta` / `p0_bg` | 1e-3 / 1e-4 | $P_0$ 的姿态、陀螺零偏标准差 |
| `p0_p` / `p0_v` / `p0_ba` | 1e-2 / 1e-2 / 5e-2 | $P_0$ 的位置、速度、加计零偏标准差 |
| `r_min` | 1e-8 | 观测 $R$ 对角线的数值下限 |
| `max_pending_frames` | 3000 | 等观测的 IMU 帧积压上限，超了队头降级为不做更新 |

两个回放节点的参数：`csv_path`（必给）、`speed`（回放倍速，时间戳保持原始）、`frame_id`。


## 数据约定

- 时间戳为同一时间基准（秒），$\Delta t$ 取相邻两帧差分；回退或重复的帧跳过并计数。
- 加速度单位为 **g**（题面约定），节点内乘 `g_ms2` 换成 m/s²，换算只在这一处发生。
- 观测与 IMU 的时间戳不同步，节点把 IMU 帧压进队列，等出现"时间戳 >= 该帧"的 pose 样本，
  再用前后两条样本 SLERP 插值到该 IMU 时刻（位置线性插值、协方差取最近邻）。直接取最近邻
  会引入半帧（最大 0.379°）的对齐误差，而 $R$ 的标准差只有 0.011°–0.029°，
  残差会被对齐误差主导。
- 姿态符号：$q$ 与 $-q$ 是同一旋转。入缓冲时按与上一条样本的点积做符号展开，否则相邻样本
  之间的 SLERP 会走 359.55° 长弧（实测毛刺在 `pose_cov.csv` 第 17957 帧，真转角只有
  0.45°）。
