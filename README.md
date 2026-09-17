# 基于 EKF 的 IMU 姿态解算

传感器组考核题目二：用 ESKF 从六轴 IMU 数据估计姿态（Roll/Pitch/Yaw），观测来自 FAST-LIO 输出的位姿。


## 构建运行

```bash
cd eskf_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

ros2 run eskf_imu imu_player_node  --ros-args -p csv_path:="$DATA/imu.csv"
ros2 run eskf_imu pose_player_node --ros-args -p csv_path:="$DATA/pose_cov.csv"
```

数据体检：

```bash
python3 eskf_ws/src/eskf_imu/scripts/data_check.py <imu.csv> <pose_cov.csv> [输出目录]
```



## 纯积分

```bash
cd eskf_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 run eskf_imu gyro_integral_node --ros-args \
  -p csv_out:=$PWD/src/eskf_imu/output/gyro_out.csv

ros2 run eskf_imu imu_player_node --ros-args -p csv_path:="$DATA/imu.csv"
```

验收：

```bash
python3 eskf_ws/src/eskf_imu/scripts/eval_gyro.py \
  eskf_ws/src/eskf_imu/output/gyro_out.csv "$DATA/pose_cov.csv" eskf_ws/src/eskf_imu/output
```

测试：

```bash
./build/eskf_imu/test_quaternion
```



## ESKF 预测步

```bash
cd eskf_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

ros2 run eskf_imu eskf_node --ros-args \
  -p csv_out:=$PWD/src/eskf_imu/output/ekf_out.csv

ros2 run eskf_imu imu_player_node --ros-args -p csv_path:="$DATA/imu.csv"
```

验收：

```bash
python3 eskf_ws/src/eskf_imu/scripts/eval_eskf_predict.py \
  eskf_ws/src/eskf_imu/output/ekf_out.csv \
  eskf_ws/src/eskf_imu/output/gyro_out.csv \
  eskf_ws/src/eskf_imu/output
```



## ESKF 观测更新

三个终端。**节点要比 player 早启动几秒**（ROS2 发现需要时间，晚订阅会丢掉开头的帧）。

```bash
cd eskf_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

# 终端 1：ESKF 节点（先起，等 /imu 与 /pose_cov）
ros2 run eskf_imu eskf_node --ros-args \
  -p csv_out:=$PWD/src/eskf_imu/output/ekf_out.csv

# 终端 2：位姿回放
ros2 run eskf_imu pose_player_node --ros-args \
  -p csv_path:="$DATA/pose_cov.csv" -p speed:=20.0

# 终端 3：IMU 回放
ros2 run eskf_imu imu_player_node --ros-args \
  -p csv_path:="$DATA/imu.csv" -p speed:=20.0
```

回放结束后 Ctrl+C 停掉终端 1，节点打印观测更新帧数、b_g 与末帧 P 对角线，并把还压在
队列里的帧 flush 掉。验收：

```bash
python3 eskf_ws/src/eskf_imu/scripts/eval_eskf_update.py \
  eskf_ws/src/eskf_imu/output/ekf_out.csv \
  "$DATA/pose_cov.csv" \
  eskf_ws/src/eskf_imu/output
```



## 一键回放脚本

三个终端手敲容易漏（尤其收尾），也可以用脚本：

```bash
cd eskf_ws/src/eskf_imu
bash scripts/run_replay.sh $PWD/output/ekf_out.csv            # 必做（6 维）
bash scripts/run_replay.sh $PWD/output/ekf_pv_out.csv -p estimate_position:=true
```

节点日志在 `<输出CSV>.log`（验收数字都在里面），脚本结束时会打印尾部。





## 评估与 Q 调参

```bash
cd eskf_ws/src/eskf_imu
# ① 题面要求的 Roll/Pitch/Yaw–Time 三条曲线 + 三方对比（图 11/12）
python3 scripts/plot.py output/ekf_out.csv output/gyro_out.csv ../../../data/pose_cov.csv output
# ② Q 扫描表 + 图 13（先跑两条回放：-p sigma_g:=0.00032 与 -p sigma_g:=0.032）
bash scripts/run_replay.sh $PWD/output/ekf_q_x0p1.csv -p sigma_g:=0.00032
bash scripts/run_replay.sh $PWD/output/ekf_q_x10.csv  -p sigma_g:=0.032
python3 scripts/q_sweep.py output/ekf_q_x0p1.csv output/ekf_out.csv output/ekf_q_x10.csv \
  ../../../data/pose_cov.csv output
# ③ 零偏初值对照 + 图 14（先跑一次 -p bg_init:=zero）
bash scripts/run_replay.sh $PWD/output/ekf_bg0.csv -p bg_init:=zero
python3 scripts/bg_init_check.py output/ekf_out.csv output/ekf_bg0.csv output
```

实测数字：EKF 与观测的测地角误差均值 0.037°（纯陀螺积分 1.100°）；σ_g 三条趋势单调
（×0.1 → 运动段 NIS 80.6、×1 → 22.5、×10 → 2.09）；零偏初值取 0 时 z 轴 6.45 s 爬到 0.0168，
两组末值逐位相同。



## 位置估计（15 维）

状态扩到 $\delta x = [\delta p, \delta v, \delta\theta, \delta b_g, \delta b_a]$，位置由加计去零偏、
去重力后双重积分预测，位置观测取自 `pose_cov.csv` 的 x,y,z（$R$ 逐帧取 cov_00/11/22）。
单位约定：加计读数 × 9.80665 → m/s²，重力向量 $(0,0,+9.80665)$。

```bash
cd eskf_ws/src/eskf_imu
source /opt/ros/humble/setup.bash && source install/setup.bash

# 主结果
bash scripts/run_replay.sh $PWD/output/ekf_pv_out.csv -p estimate_position:=true
# 对照 A：关掉位置观测（只预测）
bash scripts/run_replay.sh $PWD/output/ekf_pv_noobs.csv \
  -p estimate_position:=true -p use_position_update:=false
# 对照 B：加计零偏初值取 0
bash scripts/run_replay.sh $PWD/output/ekf_pv_ba0.csv \
  -p estimate_position:=true -p acc_bias_init:=zero

# 验收（六条判据 + 图 15–19）
python3 scripts/eval_eskf_pv.py output/ekf_pv_out.csv ../../../data/pose_cov.csv output
# 两组对照实验（图 20）
python3 scripts/eval_eskf_pv_compare.py output/ekf_pv_out.csv output/ekf_pv_noobs.csv \
  output/ekf_pv_ba0.csv ../../../data/pose_cov.csv output
```

实测数字：位置跟踪误差 RMS 1.01/1.35/0.76 mm（与观测自身噪声同量级），速度 |v|max 0.970 m/s
（观测平滑差分 0.963），加计零偏末值 (−0.0395, +0.0049, −0.0573) m/s²；
关掉位置观测那组 128 s 飘到 **801 m**（真实轨迹 9 m），加计零偏初值取 0 那组末值与主结果
**逐位相同**。









