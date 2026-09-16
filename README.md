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








