#!/usr/bin/env bash
# 回放一遍：起 eskf_node（先起，等话题）+ pose/imu 两个 player，回放完给节点发 SIGINT。
# 注意：不能开 set -u —— ROS2 的 setup.bash 自己会引用未定义变量（AMENT_TRACE_SETUP_FILES）。
set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../eskf_imu/scripts
PKG=$(cd "$HERE/.." && pwd)                          # .../eskf_imu
WS=$(cd "$PKG/../.." && pwd)                         # .../eskf_ws
ROOT=$(cd "$WS/.." && pwd)                           # .../EKF_IMU_demo
DATA="$ROOT/data"

if [ $# -lt 1 ]; then
  usage
  exit 2
fi
OUT=$1
shift
LOG="$OUT.log"
mkdir -p "$(dirname "$OUT")"

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
# 不会 flush，CSV 少尾）。所以这里按可执行文件名找真正的节点进程。
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
