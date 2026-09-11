#include <chrono>
#include <cmath>
#include <string>
#include <thread>

#include "eskf_imu/csv_reader.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

// 把 imu.csv 按原始时间戳回放到 /imu（sensor_msgs/Imu）。
// 加速度保持 g 单位；回放按真实时间间隔，不做匀速化。
class ImuPlayerNode : public rclcpp::Node {
 public:
  ImuPlayerNode() : Node("imu_player_node") {
    csv_path_ = declare_parameter<std::string>("csv_path", "");
    speed_ = declare_parameter<double>("speed", 1.0);
    accel_to_mps2_ = declare_parameter<bool>("accel_to_mps2", false);
    bias_window_sec_ = declare_parameter<double>("bias_window_sec", 60.0);
    frame_id_ = declare_parameter<std::string>("frame_id", "imu_link");

    if (csv_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "必须给参数 csv_path（imu.csv 的绝对路径）");
      throw std::runtime_error("csv_path is required");
    }

    table_ = std::make_shared<const eskf_imu::CsvTable>(eskf_imu::read_csv(csv_path_));
    const auto& t = *table_;
    RCLCPP_INFO(get_logger(), "读入 %zu 行（跳过坏行 %zu），列数 %zu",
                t.rows.size(), t.skipped_rows, t.num_cols());

    for (size_t i = 0; i < 3 && i < t.rows.size(); ++i) {
      const auto& r = t.row(i);
      RCLCPP_INFO(get_logger(),
                  "第 %zu 行: t=%.9f acc=(%.6f, %.6f, %.6f) gyro=(%.6f, %.6f, %.6f)",
                  i, r[0], r[1], r[2], r[3], r[4], r[5], r[6]);
    }

    // 静止段陀螺均值就是零偏。z 轴应该接近 0.0166，不对说明列读错了。
    const double t0 = t.row(0)[0];
    double sum[3] = {0.0, 0.0, 0.0};
    size_t n = 0;
    for (const auto& r : t.rows) {
      if (r[0] - t0 >= bias_window_sec_) break;
      for (int k = 0; k < 3; ++k) sum[k] += r[4 + k];
      ++n;
    }
    if (n > 0) {
      RCLCPP_INFO(get_logger(),
                  "静止段（前 %zu 帧, t < %.0f s）陀螺零偏估计 b_g = (%.5f, %.5f, "
                  "%.5f) rad/s",
                  n, bias_window_sec_, sum[0] / n, sum[1] / n, sum[2] / n);
    }

    pub_ = create_publisher<sensor_msgs::msg::Imu>("/imu", 50);
  }

  void run() {
    const auto& t = *table_;
    const double t0 = t.row(0)[0];
    const auto start = std::chrono::steady_clock::now();
    size_t published = 0;
    for (size_t i = 0; i < t.rows.size(); ++i) {
      if (!rclcpp::ok()) break;
      const auto& r = t.row(i);
      const auto target =
          start + std::chrono::duration<double>((r[0] - t0) / speed_);
      std::this_thread::sleep_until(target);

      sensor_msgs::msg::Imu msg;
      msg.header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(r[0] * 1e9)));
      msg.header.frame_id = frame_id_;
      const double scale = accel_to_mps2_ ? 9.80665 : 1.0;
      msg.linear_acceleration.x = r[1] * scale;
      msg.linear_acceleration.y = r[2] * scale;
      msg.linear_acceleration.z = r[3] * scale;
      msg.angular_velocity.x = r[4];
      msg.angular_velocity.y = r[5];
      msg.angular_velocity.z = r[6];
      // 姿态由 ESKF 估计，这里不填
      pub_->publish(msg);
      ++published;
    }
    RCLCPP_INFO(get_logger(), "imu.csv 回放结束：发布 %zu / %zu 帧%s",
                published, t.rows.size(),
                published == t.rows.size() ? "" : "（被中断）");
  }

 private:
  std::string csv_path_;
  double speed_ = 1.0;
  bool accel_to_mps2_ = false;
  double bias_window_sec_ = 60.0;
  std::string frame_id_;
  std::shared_ptr<const eskf_imu::CsvTable> table_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImuPlayerNode>();
  node->run();  // 回放是严格的时序循环，不走 executor 回调
  rclcpp::shutdown();
  return 0;
}
