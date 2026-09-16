#include <cmath>
#include <cstddef>
#include <deque>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "eskf_imu/eskf.hpp"
#include "eskf_imu/pose_align.hpp"

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

// M3 + M4：ESKF 节点。骨架与 gyro_integral_node 相同（同一套初始化、同一个跳帧规则），
// 姿态推进换成 Eskf::predict，M4 再订阅 /pose_cov 做观测更新。
//
// 观测与 IMU 的时间戳不同步：pose_cov.csv 起始比 imu.csv 晚 0.17 s，且两个 player
// 各自按自己的墙钟起播，"谁在前"取决于终端启动顺序。所以 IMU 帧先进 pending_ 队列，
// 等出现「时间戳 >= 该帧」的 pose 样本才处理：用前后两条 pose 样本 SLERP 插值到该
// IMU 时刻（协方差取最近邻）。直接取最近邻会引入半帧（最大 0.379°）的对齐误差，
// 而 R 的标准差只有 0.011°–0.029°，差 15 倍以上，残差会被对齐误差主导。
//
// 三道保险保证不死锁、不丢行：
//   ① 队列深度按时间戳自适应（不需要知道 0.17 s 这个数，与启动顺序无关）
//   ② pending 超过 max_pending_frames 时队头降级为「不做更新」放行（/pose_cov 没在发也能出完整 CSV）
//   ③ Ctrl+C 后 finish() 把队列里剩余的帧全部放行
class EskfNode : public rclcpp::Node {
 public:
  EskfNode() : Node("eskf_node") {
    csv_out_ = declare_parameter<std::string>("csv_out", "output/ekf_out.csv");
    bias_window_frames_ = declare_parameter<int>("bias_window_frames", 1000);
    sigma_g_ = declare_parameter<double>("sigma_g", 0.0032);
    sigma_bg_ = declare_parameter<double>("sigma_bg", 1e-4);
    p0_theta_ = declare_parameter<double>("p0_theta", 1e-3);  // P0 = diag(1e-6 I, 1e-8 I)
    p0_bg_ = declare_parameter<double>("p0_bg", 1e-4);
    use_pose_update_ = declare_parameter<bool>("use_pose_update", true);
    r_min_ = declare_parameter<double>("r_min", 1e-8);
    max_pending_frames_ = declare_parameter<int>("max_pending_frames", 3000);

    auto qos = rclcpp::QoS(20000);  // 25613 帧，回放时不丢帧
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", qos, [this](const sensor_msgs::msg::Imu::SharedPtr msg) { on_imu(msg); });
    if (use_pose_update_) {
      sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "/pose_cov", qos,
          [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
            on_pose(msg);
          });
    } else {
      RCLCPP_WARN(get_logger(),
                  "use_pose_update=false：不订阅 /pose_cov，本节点退化为 M3 的纯预测（带 P）");
    }
  }

  void finish() {
    if (!started_ && !buffer_.empty()) {
      start();  // 缓冲未满也要出结果
    }
    // 还压在队列里的帧全部放行（不再等括号观测），保证行数与纯陀螺积分一致
    while (!pending_.empty()) {
      Obs obs;
      step(pending_.front(), obs);
      pending_.pop_front();
      ++flushed_;
    }
    if (fout_.is_open()) {
      fout_.close();
      RCLCPP_INFO(get_logger(),
                  "ESKF 结束：写出 %zu 帧（dt 异常跳过 %zu），其中观测更新 %zu 帧、无观测 %zu 帧"
                  "（队列上限兜底 %zu、结束 flush %zu）",
                  written_, skipped_dt_, updated_, no_obs_, stale_drained_, flushed_);
      RCLCPP_INFO(get_logger(),
                  "收到 /pose_cov %zu 帧（开头协方差全零 %zu 帧），姿态观测乱序丢弃 %zu 帧",
                  pose_msgs_, zero_cov_pose_, pose_align_.out_of_order());
      RCLCPP_INFO(get_logger(), "b_g = (%.6f, %.6f, %.6f) rad/s，末帧 P 对角线 = (%.3e, %.3e, "
                                "%.3e, %.3e, %.3e, %.3e)",
                  bg_[0], bg_[1], bg_[2], last_p_[0], last_p_[1], last_p_[2], last_p_[3],
                  last_p_[4], last_p_[5]);
      if (use_pose_update_ && pose_msgs_ == 0) {
        RCLCPP_WARN(get_logger(),
                    "一帧 /pose_cov 都没收到：确认 pose_player_node 在跑，否则整段没有观测修正");
      }
    }
  }

 private:
  struct Sample {
    double t;
    double gyro[3];
    double acc[3];
  };

  struct Obs {
    bool have = false;
    eskf_imu::Quat q;
    double r[3] = {0.0, 0.0, 0.0};
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
        start();
      }
      return;
    }
    pending_.push_back(s);
    drain();
  }

  // 位姿观测入缓冲。协方差整行 36 项全零 = FAST-LIO 尚未初始化，标记为不可用。
  void on_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    // 6×6 协方差按行主序摊成 36 个元素。题面说的 cov_33/44/55 是 **CSV 列名**
    // （第 3/4/5 行第 3/4/5 列，即 roll/pitch/yaw 的方差），摊平后下标是 3*6+3=21、
    // 4*6+4=28、5*6+5=35 —— 不是 33/44/55（那两个超出 36 个元素的范围，会读到别的内存）。
    static const int kCovIdx[3] = {21, 28, 35};
    double r[3] = {msg->pose.covariance[kCovIdx[0]], msg->pose.covariance[kCovIdx[1]],
                   msg->pose.covariance[kCovIdx[2]]};

    bool all_zero = true;
    for (int k = 0; k < 36; ++k) {
      if (msg->pose.covariance[k] != 0.0) {
        all_zero = false;
        break;
      }
    }
    if (all_zero) ++zero_cov_pose_;
    ++pose_msgs_;

    // 量级自检：实测 cov_33/44/55 落在 4e-8 ~ 3e-7（plan.md §1.5）。取错列时这里会报。
    if (!all_zero && !checked_r_range_) {
      checked_r_range_ = true;
      const bool in_range = (r[0] > 1e-9 && r[0] < 1e-5) && (r[1] > 1e-9 && r[1] < 1e-5) &&
                            (r[2] > 1e-9 && r[2] < 1e-5);
      if (in_range) {
        RCLCPP_INFO(get_logger(), "首个可用观测的 R 对角线 = (%.3e, %.3e, %.3e) rad²（实测范围 4e-8~3e-7）",
                    r[0], r[1], r[2]);
      } else {
        RCLCPP_WARN(get_logger(),
                    "R 对角线 = (%.3e, %.3e, %.3e) 超出实测范围 4e-8~3e-7：很可能协方差列取错"
                    "（cov_33/44/55 摊平后是 21/28/35）",
                    r[0], r[1], r[2]);
      }
    }

    const auto& o = msg->pose.pose.orientation;
    pose_align_.push(t, {o.w, o.x, o.y, o.z}, r[0], r[1], r[2], !all_zero);
    drain();
  }

  // 初始化与 gyro_integral_node 完全一致
  void start() {
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
    eskf_.set_r_min(r_min_);
    eskf_.init(eskf_imu::quat_from_euler_zyx(e0.roll, e0.pitch, e0.yaw), bg0, p0_theta_,
               p0_bg_);

    fout_.open(csv_out_);
    if (!fout_) {
      RCLCPP_FATAL(get_logger(), "无法打开输出文件: %s", csv_out_.c_str());
      throw std::runtime_error("csv_out open failed");
    }
    fout_ << "time,dt,qw,qx,qy,qz,roll_deg,pitch_deg,yaw_deg,"
             "bg_x,bg_y,bg_z,p00,p11,p22,p33,p44,p55,"
             "z_x,z_y,z_z,nis,S00,S11,S22,upd\n";  // 角度：度；P/S：对角线；upd：本帧是否做了观测更新

    for (size_t i = 0; i < n; ++i) {
      pending_.push_back(buffer_[i]);
    }
    buffer_.clear();
    started_ = true;
    drain();
  }

  // 队头帧能否处理：能就填好 obs 并返回 true，不能（要等未来的姿态样本）返回 false。
  bool resolve_obs(double t, Obs& obs) {
    obs.have = false;
    if (!use_pose_update_) return true;  // 观测通道整体关闭，直接放行

    eskf_imu::PoseObs po;
    switch (pose_align_.lookup(t, po)) {
      case eskf_imu::PoseAligner::kOk:
        if (po.valid) {
          obs.have = true;
          obs.q = po.q;
          for (int k = 0; k < 3; ++k) obs.r[k] = po.r_diag[k];
        }
        return true;
      case eskf_imu::PoseAligner::kNoObsYet:
        // t 早于最早的姿态样本（实测前 29 个 IMU 帧），永远等不到括号：无观测放行
        return true;
      case eskf_imu::PoseAligner::kNeedMore:
      default:
        return false;
    }
  }

  // 按队列顺序（= 时间戳顺序）处理，能处理多少处理多少
  void drain() {
    while (!pending_.empty()) {
      Obs obs;
      if (!resolve_obs(pending_.front().t, obs)) {
        if (pending_.size() <= static_cast<size_t>(max_pending_frames_)) return;
        if (!warned_stale_) {
          warned_stale_ = true;
          RCLCPP_WARN(get_logger(),
                      "pending 已积压 %zu 帧仍等不到带括号的姿态观测：确认 /pose_cov 在发布、"
                      "pose_player_node 没有落后太多。队头帧将降级为不做更新放行。",
                      pending_.size());
        }
        ++stale_drained_;
      }
      step(pending_.front(), obs);
      pending_.pop_front();
    }
  }

  void step(const Sample& s, const Obs& obs) {
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

    double z[3] = {0, 0, 0}, sd[3] = {0, 0, 0}, nis = 0.0;
    if (obs.have) {
      const eskf_imu::UpdateResult r = eskf_.update(obs.q, obs.r);
      for (int k = 0; k < 3; ++k) {
        z[k] = r.z[k];
        sd[k] = r.s_diag[k];
      }
      nis = r.nis;
      ++updated_;
    } else {
      ++no_obs_;
    }

    const eskf_imu::Quat& q = eskf_.q();
    const double* bg = eskf_.bg();
    const eskf_imu::Mat6& P = eskf_.P();
    for (int i = 0; i < 3; ++i) {
      bg_[i] = bg[i];
      last_p_[i] = P(i, i);
      last_p_[3 + i] = P(3 + i, 3 + i);
    }

    const eskf_imu::EZYX e = eskf_imu::quat_to_euler_zyx(q);
    // 分组设置精度：时间戳要保留 ns（fixed 9），四元数 ~1 需 9 位小数；
    // P / S / bg / 残差都是极小量（P0=1e-8、S≈1e-7），必须用科学计数法，
    // 否则 fixed 默认 6 位会把它们统统写成 0。
    fout_ << std::fixed << std::setprecision(9) << s.t << std::setprecision(6) << "," << dt;
    fout_ << std::setprecision(9) << "," << q.w << "," << q.x << "," << q.y << "," << q.z;
    fout_ << std::setprecision(6) << "," << e.roll * 180.0 / M_PI << "," << e.pitch * 180.0 / M_PI
          << "," << e.yaw * 180.0 / M_PI;
    fout_ << std::scientific << std::setprecision(6) << "," << bg[0] << "," << bg[1] << ","
          << bg[2];
    for (int i = 0; i < 6; ++i) fout_ << "," << last_p_[i];
    fout_ << "," << z[0] << "," << z[1] << "," << z[2];
    fout_ << std::fixed << std::setprecision(6) << "," << nis;
    fout_ << std::scientific << std::setprecision(6) << "," << sd[0] << "," << sd[1] << ","
          << sd[2] << "," << (obs.have ? 1 : 0) << "\n";
    ++written_;
  }

  std::string csv_out_;
  int bias_window_frames_ = 1000;
  double sigma_g_ = 0.0032, sigma_bg_ = 1e-4;
  double p0_theta_ = 1e-3, p0_bg_ = 1e-4;
  bool use_pose_update_ = true;
  double r_min_ = 1e-8;
  int max_pending_frames_ = 3000;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;

  eskf_imu::Eskf eskf_;
  eskf_imu::PoseAligner pose_align_;
  std::vector<Sample> buffer_;      // 初始化窗口（前 bias_window_frames 帧）
  std::deque<Sample> pending_;      // 等带括号姿态观测的 IMU 帧
  bool started_ = false;
  bool warned_stale_ = false;
  bool checked_r_range_ = false;
  bool have_last_t_ = false;
  double last_t_ = 0.0;
  double bg_[3] = {0, 0, 0};
  double last_p_[6] = {0, 0, 0, 0, 0, 0};
  std::ofstream fout_;
  size_t written_ = 0, skipped_dt_ = 0, updated_ = 0, no_obs_ = 0;
  size_t stale_drained_ = 0, flushed_ = 0, pose_msgs_ = 0, zero_cov_pose_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EskfNode>();
  rclcpp::spin(node);
  node->finish();
  rclcpp::shutdown();
  return 0;
}
