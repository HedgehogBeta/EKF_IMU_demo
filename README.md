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





