#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "eskf_imu/eskf.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

class EskfNode : public rclcpp::Node {
 public:
  EskfNode() : Node("eskf_node") {
    csv_out_ = declare_parameter<std::string>("csv_out", "output/ekf_out.csv");
    bias_window_frames_ = declare_parameter<int>("bias_window_frames", 1000);
    sigma_g_ = declare_parameter<double>("sigma_g", 0.0032);
    sigma_bg_ = declare_parameter<double>("sigma_bg", 1e-4);
    p0_theta_ = declare_parameter<double>("p0_theta", 1e-3);  // P0 = diag(1e-6 I, 1e-8 I)
    p0_bg_ = declare_parameter<double>("p0_bg", 1e-4);

    auto qos = rclcpp::QoS(20000);  // 25613 帧，回放时不丢帧
    sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", qos, [this](const sensor_msgs::msg::Imu::SharedPtr msg) { on_imu(msg); });
  }

  void finish() {
    if (!started_ && !buffer_.empty()) {
      start_predict();
    }
    if (fout_.is_open()) {
      fout_.close();
      RCLCPP_INFO(get_logger(),
                  "ESKF 预测步结束：写出 %zu 帧（dt 异常跳过 %zu），b_g = (%.5f, %.5f, %.5f) "
                  "rad/s，末帧 P 对角线 = (%.3e, %.3e, %.3e, %.3e, %.3e, %.3e)",
                  written_, skipped_dt_, bg_[0], bg_[1], bg_[2], last_p_[0], last_p_[1],
                  last_p_[2], last_p_[3], last_p_[4], last_p_[5]);
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
        start_predict();
      }
      return;
    }
    step(s);
  }

  // 初始化与 gyro_integral_node 完全一致
  void start_predict() {
    const size_t n = buffer_.size();
    double bg0[3] = {0, 0, 0}, fa[3] = {0, 0, 0};
    for (const auto& s : buffer_) {
      for (int k = 0; k < 3; ++k) {
        bg0[k] += s.gyro[k];
        fa[k] += s.acc[k];
      }
    }
    for (int k = 0; k < 3; ++k) {
      bg0[k] /= static_cast<double>(n);
      fa[k] /= static_cast<double>(n);
    }

    const eskf_imu::EZYX e0 = eskf_imu::euler_from_specific_force({fa[0], fa[1], fa[2]});
    eskf_.set_noise(sigma_g_, sigma_bg_);
    eskf_.init(eskf_imu::quat_from_euler_zyx(e0.roll, e0.pitch, e0.yaw), bg0, p0_theta_,
               p0_bg_);

    fout_.open(csv_out_);
    if (!fout_) {
      RCLCPP_FATAL(get_logger(), "无法打开输出文件: %s", csv_out_.c_str());
      throw std::runtime_error("csv_out open failed");
    }
    fout_ << "time,dt,qw,qx,qy,qz,roll_deg,pitch_deg,yaw_deg,"
             "bg_x,bg_y,bg_z,p00,p11,p22,p33,p44,p55\n";  // 角度：度；P：对角线

    for (size_t i = 0; i < n; ++i) {
      step(buffer_[i]);
    }
    buffer_.clear();
    started_ = true;
  }

  void step(const Sample& s) {
    double dt = 0.0;
    if (have_last_t_) {
      dt = s.t - last_t_;
      if (dt <= 0.0) {  // 与 gyro_integral_node 相同的跳过规则，保证逐帧对齐
        ++skipped_dt_;
        last_t_ = s.t;
        return;
      }
      eskf_.predict(dt, s.gyro);
    }
    have_last_t_ = true;
    last_t_ = s.t;

    const eskf_imu::Quat& q = eskf_.q();
    const double* bg = eskf_.bg();
    const eskf_imu::Mat6& P = eskf_.P();
    for (int i = 0; i < 3; ++i) {
      bg_[i] = bg[i];
      last_p_[i] = P(i, i);
      last_p_[3 + i] = P(3 + i, 3 + i);
    }

    const eskf_imu::EZYX e = eskf_imu::quat_to_euler_zyx(q);
    fout_ << std::fixed << s.t << "," << dt << "," << q.w << "," << q.x << "," << q.y << ","
          << q.z << "," << e.roll * 180.0 / M_PI << "," << e.pitch * 180.0 / M_PI << ","
          << e.yaw * 180.0 / M_PI << "," << bg[0] << "," << bg[1] << "," << bg[2] << ","
          << last_p_[0] << "," << last_p_[1] << "," << last_p_[2] << "," << last_p_[3] << ","
          << last_p_[4] << "," << last_p_[5] << "\n";
    ++written_;
  }

  std::string csv_out_;
  int bias_window_frames_ = 1000;
  double sigma_g_ = 0.0032, sigma_bg_ = 1e-4;
  double p0_theta_ = 1e-3, p0_bg_ = 1e-4;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;

  eskf_imu::Eskf eskf_;
  std::vector<Sample> buffer_;
  bool started_ = false;
  bool have_last_t_ = false;
  double last_t_ = 0.0;
  double bg_[3] = {0, 0, 0};
  double last_p_[6] = {0, 0, 0, 0, 0, 0};
  std::ofstream fout_;
  size_t written_ = 0, skipped_dt_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EskfNode>();
  rclcpp::spin(node);
  node->finish();
  rclcpp::shutdown();
  return 0;
}
