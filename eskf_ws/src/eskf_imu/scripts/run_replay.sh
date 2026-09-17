#!/usr/bin/env bash
# 回放一遍：起 eskf_node（先起，等话题）+ pose/imu 两个 player，回放完给节点发 SIGINT。
#
# 用法：
#   scripts/run_replay.sh <输出CSV绝对路径> [eskf_node 的额外 -p 参数 ...]
#
# 例：
#   scripts/run_replay.sh $PWD/output/ekf_out.csv
#   scripts/run_replay.sh $PWD/output/ekf_pv_out.csv -p estimate_position:=true
#   scripts/run_replay.sh $PWD/output/ekf_pv_noobs.csv -p estimate_position:=true -p use_position_update:=false
#
# 约定与 M4 手跑一致：节点比 player 早 3 s 启动（ROS2 发现需要时间，晚订阅会丢开头的帧）；
# 两个 player 都按 speed:=20 加速回放，但**时间戳仍是原始的**，Δt 与观测对齐逻辑不受影响。
# 节点日志留在 <输出CSV>.log（里面有验收数字）。
# 注意：不能开 set -u —— ROS2 的 setup.bash 自己会引用未定义变量（AMENT_TRACE_SETUP_FILES）。
set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../eskf_imu/scripts
PKG=$(cd "$HERE/.." && pwd)                          # .../eskf_imu
WS=$(cd "$PKG/../.." && pwd)                         # .../eskf_ws
ROOT=$(cd "$WS/.." && pwd)                           # .../EKF_IMU_demo
DATA="$ROOT/data"

if [ $# -lt 1 ]; then
  sed -n '2,12p' "$0"
  exit 2
fi
OUT=$1
shift
LOG="$OUT.log"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

echo "输出: $OUT"
echo "日志: $LOG"
echo "节点参数: $*"

ros2 run eskf_imu eskf_node --ros-args -p csv_out:="$OUT" "$@" >"$LOG" 2>&1 &
NODE=$!
sleep 3

ros2 run eskf_imu pose_player_node --ros-args \
  -p csv_path:="$DATA/pose_cov.csv" -p speed:=20.0 >"$LOG.pose" 2>&1 &
POSE=$!
sleep 0.5

ros2 run eskf_imu imu_player_node --ros-args \
  -p csv_path:="$DATA/imu.csv" -p speed:=20.0 >"$LOG.imu" 2>&1

wait $POSE 2>/dev/null
sleep 0.5

# 收尾必须给**节点本身**发 SIGINT：`ros2 run` 是 python 包装进程，真正的节点是它的子进程，
# 只 kill 包装进程打不到节点，节点就不会走 finish()（还压在 pending_ 队列里的最后几十帧
# 不会 flush，CSV 少尾）。所以这里给整个进程组发信号。
# （终端里手跑时按 Ctrl+C 是对整个前台进程组发 SIGINT，所以一直没暴露这个问题。）
NODE_PID=$(pgrep -f "lib/eskf_imu/eskf_node" | head -1)
if [ -n "${NODE_PID:-}" ]; then
  kill -INT "$NODE_PID"
else
  kill -INT "$NODE" 2>/dev/null
fi
for _ in $(seq 1 20); do
  pgrep -f "lib/eskf_imu/eskf_node" >/dev/null || break
  sleep 0.5
done
pkill -TERM -f "lib/eskf_imu/eskf_node" 2>/dev/null
wait $NODE 2>/dev/null

echo "---- 节点日志（尾部）----"
tail -n 12 "$LOG"
