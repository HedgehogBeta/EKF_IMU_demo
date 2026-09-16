#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "eskf_imu/quaternion.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

// M2：纯陀螺积分节点。订阅 /imu，先缓冲静止段前 N 帧估计零偏 b_g 与初始姿态
//（加速度计解算 roll/pitch，yaw=0），然后 q <- q ⊗ Exp((ω_m - b_g)·Δt) 逐帧积分，
// Δt 取消息时间戳差分（含启动抖动，不是固定 1/200 s），每帧归一化，RPY 写 CSV。
class GyroIntegralNode : public rclcpp::Node {
 public:
  GyroIntegralNode() : Node("gyro_integral_node") {
    csv_out_ = declare_parameter<std::string>("csv_out", "output/gyro_out.csv");
    bias_window_frames_ = declare_parameter<int>("bias_window_frames", 1000);

    auto qos = rclcpp::QoS(20000);  // 25613 帧，回放时不丢帧
    sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", qos, [this](const sensor_msgs::msg::Imu::SharedPtr msg) { on_imu(msg); });
  }

  // 供 main 在 spin 退出后收尾：缓冲未满也要出结果，CSV 无论如何落盘
  void finish() {
    if (!started_ && !buffer_.empty()) {
      start_integration();
    }
    if (fout_.is_open()) {
      fout_.close();
      RCLCPP_INFO(get_logger(),
                  "纯陀螺积分结束：写出 %zu 帧（dt 异常跳过 %zu），b_g = (%.5f, %.5f, %.5f) "
                  "rad/s，初始 roll = %.3f°，pitch = %.3f°（参考 −0.197° / +0.546°）",
                  written_, skipped_dt_, bg_[0], bg_[1], bg_[2], init_roll_deg_, init_pitch_deg_);
    }
  }

 private:
  struct Sample {
    double t;
    double gyro[3];
    double acc[3];
  };

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    Sample s;
    s.t = rclcpp::Time(msg->header.stamp).seconds();
    s.gyro[0] = msg->angular_velocity.x;
    s.gyro[1] = msg->angular_velocity.y;
    s.gyro[2] = msg->angular_velocity.z;
    s.acc[0] = msg->linear_acceleration.x;  // g 单位（M1 保持，未换算）
    s.acc[1] = msg->linear_acceleration.y;
    s.acc[2] = msg->linear_acceleration.z;

    if (!started_) {
      buffer_.push_back(s);
      if (static_cast<int>(buffer_.size()) >= bias_window_frames_) {
        start_integration();
      }
      return;
    }
    integrate(s);
  }

  // 静止段均值 = 零偏；比力均值方向 = 初始 roll/pitch。之后从第 0 帧起把缓冲整段积分补上。
  void start_integration() {
    const size_t n = buffer_.size();
    double bg[3] = {0, 0, 0}, fa[3] = {0, 0, 0};
    for (const auto& s : buffer_) {
      for (int k = 0; k < 3; ++k) {
        bg[k] += s.gyro[k];
        fa[k] += s.acc[k];
      }
    }
    for (int k = 0; k < 3; ++k) {
      bg_[k] = bg[k] / static_cast<double>(n);
      fa_[k] = fa[k] / static_cast<double>(n);
    }

    const eskf_imu::EZYX e0 =
        eskf_imu::euler_from_specific_force({fa_[0], fa_[1], fa_[2]});
    q_ = eskf_imu::quat_from_euler_zyx(e0.roll, e0.pitch, e0.yaw);
    init_roll_deg_ = e0.roll * 180.0 / M_PI;
    init_pitch_deg_ = e0.pitch * 180.0 / M_PI;

    fout_.open(csv_out_);
    if (!fout_) {
      RCLCPP_FATAL(get_logger(), "无法打开输出文件: %s", csv_out_.c_str());
      throw std::runtime_error("csv_out open failed");
    }
    fout_ << "time,dt,qw,qx,qy,qz,roll_deg,pitch_deg,yaw_deg\n";  // RPY 单位：度

    for (size_t i = 0; i < n; ++i) {
      integrate(buffer_[i]);
    }
    buffer_.clear();
    started_ = true;
  }

  void integrate(const Sample& s) {
    double dt = 0.0;
    if (have_last_t_) {
      dt = s.t - last_t_;
      if (dt <= 0.0) {  // 时间戳回退/重复：没法积分，跳过并计数
        ++skipped_dt_;
        last_t_ = s.t;
        return;
      }
      const double w[3] = {s.gyro[0] - bg_[0], s.gyro[1] - bg_[1], s.gyro[2] - bg_[2]};
      q_ = eskf_imu::qnormalize(
          eskf_imu::qmul(q_, eskf_imu::qexp({w[0] * dt, w[1] * dt, w[2] * dt})));
    }
    have_last_t_ = true;
    last_t_ = s.t;

    const eskf_imu::EZYX e = eskf_imu::quat_to_euler_zyx(q_);
    // 时间戳 fixed 9 保留 ns（与 eskf_node 同格式，否则两份 CSV 的时间戳对不齐），
    // 四元数 9 位小数，dt 与角度 6 位
    fout_ << std::fixed << std::setprecision(9) << s.t << std::setprecision(6) << "," << dt;
    fout_ << std::setprecision(9) << "," << q_.w << "," << q_.x << "," << q_.y << "," << q_.z;
    fout_ << std::setprecision(6) << "," << e.roll * 180.0 / M_PI << ","
          << e.pitch * 180.0 / M_PI << "," << e.yaw * 180.0 / M_PI << "\n";
    ++written_;
  }

  std::string csv_out_;
  int bias_window_frames_ = 1000;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;

  std::vector<Sample> buffer_;
  bool started_ = false;
  bool have_last_t_ = false;
  double last_t_ = 0.0;
  double bg_[3] = {0, 0, 0};
  double fa_[3] = {0, 0, 0};
  double init_roll_deg_ = 0.0, init_pitch_deg_ = 0.0;
  eskf_imu::Quat q_;
  std::ofstream fout_;
  size_t written_ = 0, skipped_dt_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GyroIntegralNode>();
  rclcpp::spin(node);  // Ctrl+C 或 shutdown 后到这里，缓冲未满也能出结果
  node->finish();
  rclcpp::shutdown();
  return 0;
}
