#include <chrono>
#include <cmath>
#include <string>
#include <thread>

#include "eskf_imu/csv_reader.hpp"

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

// 把 pose_cov.csv（FAST-LIO 的位姿，作为观测）按原始时间戳回放到 /pose_cov。
// 四元数列是 qx,qy,qz,qw（标量最后），协方差 36 列按行主序直接拷贝。
class PosePlayerNode : public rclcpp::Node {
 public:
  PosePlayerNode() : Node("pose_player_node") {
    csv_path_ = declare_parameter<std::string>("csv_path", "");
    speed_ = declare_parameter<double>("speed", 1.0);
    frame_id_ = declare_parameter<std::string>("frame_id", "map");

    if (csv_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "必须给参数 csv_path（pose_cov.csv 的绝对路径）");
      throw std::runtime_error("csv_path is required");
    }

    table_ = std::make_shared<const eskf_imu::CsvTable>(eskf_imu::read_csv(csv_path_));
    const auto& t = *table_;
    RCLCPP_INFO(get_logger(), "读入 %zu 行（跳过 %zu），起始时刻 %.9f",
                t.rows.size(), t.skipped_rows, t.row(0)[0]);

    pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_cov", 50);
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

      geometry_msgs::msg::PoseWithCovarianceStamped msg;
      msg.header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(r[0] * 1e9)));
      msg.header.frame_id = frame_id_;
      msg.pose.pose.position.x = r[1];
      msg.pose.pose.position.y = r[2];
      msg.pose.pose.position.z = r[3];
      msg.pose.pose.orientation.x = r[4];
      msg.pose.pose.orientation.y = r[5];
      msg.pose.pose.orientation.z = r[6];
      msg.pose.pose.orientation.w = r[7];
      for (int k = 0; k < 36; ++k) {
        msg.pose.covariance[k] = r[8 + k];
      }
      pub_->publish(msg);
      ++published;
    }
    RCLCPP_INFO(get_logger(), "pose_cov.csv 回放结束：发布 %zu / %zu 帧%s",
                published, t.rows.size(),
                published == t.rows.size() ? "" : "（被中断）");
  }

 private:
  std::string csv_path_;
  double speed_ = 1.0;
  std::string frame_id_;
  std::shared_ptr<const eskf_imu::CsvTable> table_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PosePlayerNode>();
  node->run();
  rclcpp::shutdown();
  return 0;
}
