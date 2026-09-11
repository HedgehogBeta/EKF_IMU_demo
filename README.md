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


