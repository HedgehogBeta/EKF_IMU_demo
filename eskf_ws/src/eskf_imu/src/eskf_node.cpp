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

// M3 + M4 + （M7/M8 选做）ESKF 节点。骨架与 gyro_integral_node 相同（同一套初始化、同一个
// 跳帧规则），姿态推进换成 Eskf::predict，M4 订阅 /pose_cov 做姿态观测更新，
// 选做再用同一条观测里的位置做位置观测更新。
//
// 观测与 IMU 的时间戳不同步：pose_cov.csv 起始比 imu.csv 晚 0.17 s，且两个 player
// 各自按自己的墙钟起播，"谁在前"取决于终端启动顺序。所以 IMU 帧先进 pending_ 队列，
// 等出现「时间戳 >= 该帧」的 pose 样本才处理：用前后两条 pose 样本 SLERP 插值到该
// IMU 时刻（位置线性插值，协方差取最近邻）。直接取最近邻会引入半帧（最大 0.379°）的
// 对齐误差，而 R 的标准差只有 0.011°–0.029°，差 15 倍以上，残差会被对齐误差主导。
//
// 三道保险保证不死锁、不丢行：
//   ① 队列深度按时间戳自适应（不需要知道 0.17 s 这个数，与启动顺序无关）
//   ② pending 超过 max_pending_frames 时队头降级为「不做更新」放行（/pose_cov 没在发也能出完整 CSV）
//   ③ Ctrl+C 后 finish() 把队列里剩余的帧全部放行
//
// 选做（位置估计）的单位约定：算法层只认 SI（m、m/s、m/s²）。实测 imu.csv 的加速度是 g，
// 所以本节点乘 g_ms2（默认 9.80665）换算成 m/s² 再交给算法 —— 这就是 plan.md §2.5 要求的
// 「二选一」，此处选「换成 m/s²」；g_ms2 同时是算法里重力向量 (0,0,+|g|) 的模长，一个参数
// 管住两处，不会出现「换算了加速度却忘了重力模长」的错配。
class EskfNode : public rclcpp::Node {
 public:
  EskfNode() : Node("eskf_node") {
    csv_out_ = declare_parameter<std::string>("csv_out", "output/ekf_out.csv");
    bias_window_frames_ = declare_parameter<int>("bias_window_frames", 1000);
    bg_init_ = declare_parameter<std::string>("bg_init", "static");
    sigma_g_ = declare_parameter<double>("sigma_g", 0.0032);
    sigma_bg_ = declare_parameter<double>("sigma_bg", 1e-4);
    p0_theta_ = declare_parameter<double>("p0_theta", 1e-3);  // P0 = diag(1e-6 I, 1e-8 I)
    p0_bg_ = declare_parameter<double>("p0_bg", 1e-4);
    use_pose_update_ = declare_parameter<bool>("use_pose_update", true);
    r_min_ = declare_parameter<double>("r_min", 1e-8);
    max_pending_frames_ = declare_parameter<int>("max_pending_frames", 3000);
    // ---- M7/M8 选做 ----
    estimate_position_ = declare_parameter<bool>("estimate_position", false);
    use_position_update_ = declare_parameter<bool>("use_position_update", true);
    g_ms2_ = declare_parameter<double>("g_ms2", 9.80665);
    acc_bias_init_ = declare_parameter<std::string>("acc_bias_init", "static");
    p0_p_ = declare_parameter<double>("p0_p", 1e-2);
    p0_v_ = declare_parameter<double>("p0_v", 1e-2);
    p0_ba_ = declare_parameter<double>("p0_ba", 5e-2);
    sigma_a_ = declare_parameter<double>("sigma_a", 0.12);
    sigma_ba_ = declare_parameter<double>("sigma_ba", 5e-4);

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
                  "use_pose_update=false：不订阅 /pose_cov，本节点退化为纯预测（带 P）");
    }
    if (estimate_position_ && !use_position_update_) {
      RCLCPP_WARN(get_logger(),
                  "use_position_update=false：位置照常预测但不接位置观测"
                  "（对照实验：位置观测对估计的作用）");
    }
    if (estimate_position_ && !use_pose_update_) {
      RCLCPP_WARN(get_logger(),
                  "estimate_position=true 且 use_pose_update=false：姿态与位置都没有观测，"
                  "位置是加计双重积分的自由积分 —— 这条曲线会飞走，属于对照实验");
    }
    if (!estimate_position_) {
      RCLCPP_INFO(get_logger(),
                  "estimate_position=false：只做必做的姿态估计（6 维误差状态的行为），"
                  "CSV 列与 M4 完全一致");
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
                  "ESKF 结束：写出 %zu 帧（dt 异常跳过 %zu），其中姿态观测更新 %zu 帧、"
                  "无姿态观测 %zu 帧（队列上限兜底 %zu、结束 flush %zu）",
                  written_, skipped_dt_, updated_, no_obs_, stale_drained_, flushed_);
      if (estimate_position_) {
        RCLCPP_INFO(get_logger(), "其中位置观测更新 %zu 帧、无位置观测 %zu 帧", updated_pos_,
                    written_ - updated_pos_);
      }
      RCLCPP_INFO(get_logger(),
                  "收到 /pose_cov %zu 帧（开头协方差全零 %zu 帧），姿态观测乱序丢弃 %zu 帧",
                  pose_msgs_, zero_cov_pose_, pose_align_.out_of_order());
      RCLCPP_INFO(get_logger(), "b_g = (%.6f, %.6f, %.6f) rad/s，末帧 P 对角线 = (%.3e, %.3e, "
                                "%.3e, %.3e, %.3e, %.3e)",
                  bg_[0], bg_[1], bg_[2], last_p_[0], last_p_[1], last_p_[2], last_p_[3],
                  last_p_[4], last_p_[5]);
      if (estimate_position_) {
        RCLCPP_INFO(get_logger(),
                    "b_a = (%.6f, %.6f, %.6f) m/s²（初值 %.6f, %.6f, %.6f），"
                    "末帧 p = (%.4f, %.4f, %.4f) m，v = (%.4f, %.4f, %.4f) m/s",
                    ba_[0], ba_[1], ba_[2], ba0_[0], ba0_[1], ba0_[2], p_[0], p_[1], p_[2],
                    v_[0], v_[1], v_[2]);
        RCLCPP_INFO(get_logger(),
                    "末帧 P 对角线（p/v/b_a 块）= (%.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, "
                    "%.3e, %.3e)",
                    last_ppv_[0], last_ppv_[1], last_ppv_[2], last_ppv_[3], last_ppv_[4],
                    last_ppv_[5], last_ppv_[6], last_ppv_[7], last_ppv_[8]);
      }
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
    double p[3] = {0.0, 0.0, 0.0};
    double r[3] = {0.0, 0.0, 0.0};
    double r_pos[3] = {0.0, 0.0, 0.0};
  };

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    Sample s;
    s.t = rclcpp::Time(msg->header.stamp).seconds();
    s.gyro[0] = msg->angular_velocity.x;
    s.gyro[1] = msg->angular_velocity.y;
    s.gyro[2] = msg->angular_velocity.z;
    // 消息里是 g 单位（M1 的约定，imu_player 的 accel_to_mps2 保持 false 即可）
    s.acc[0] = msg->linear_acceleration.x;
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
    // 位置同理：cov_00/11/22 是位置 x,y,z 的方差，摊平后是 0、7、14。
    static const int kCovPosIdx[3] = {0, 7, 14};
    double r[3] = {0.0, 0.0, 0.0};
    double r_pos[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
      r[k] = msg->pose.covariance[kCovIdx[k]];
      r_pos[k] = msg->pose.covariance[kCovPosIdx[k]];
    }

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
      if (estimate_position_) {
        const bool pos_in_range = (r_pos[0] > 1e-8 && r_pos[0] < 1e-4) &&
                                  (r_pos[1] > 1e-8 && r_pos[1] < 1e-4) &&
                                  (r_pos[2] > 1e-8 && r_pos[2] < 1e-4);
        if (pos_in_range) {
          RCLCPP_INFO(get_logger(),
                      "首个可用观测的位置 R 对角线 = (%.3e, %.3e, %.3e) m²（实测范围 7.7e-7~2.6e-6）",
                      r_pos[0], r_pos[1], r_pos[2]);
        } else {
          RCLCPP_WARN(get_logger(),
                      "位置 R 对角线 = (%.3e, %.3e, %.3e) 超出实测范围 7.7e-7~2.6e-6：很可能取错列"
                      "（cov_00/11/22 摊平后是 0/7/14）",
                      r_pos[0], r_pos[1], r_pos[2]);
        }
      }
    }

    const auto& o = msg->pose.pose.orientation;
    const auto& pos = msg->pose.pose.position;
    const double p_obs[3] = {pos.x, pos.y, pos.z};
    pose_align_.push(t, {o.w, o.x, o.y, o.z}, p_obs, r, r_pos, !all_zero);
    drain();
  }

  // 初始化与 gyro_integral_node 完全一致（前 bias_window_frames 帧均值）：
  //   陀螺均值 → b_g；比力均值方向 → 初始 roll/pitch（yaw=0）。
  // 选做再加一条：加计零偏 b_a = |g|·(f̄ − R₀ᵀ·(0,0,1))。
  //   静止时比力等于重力在该姿态下的理论投影，实测比力减去它就是零偏。
  //   实测 f̄ 的模长只有 0.9943（差 0.57%），这一项几乎全被 z 轴吸收
  //   （≈ 0.0058 g = 0.057 m/s²）—— 它其实是加计的尺度因子误差，这里按零偏建模，
  //   是选做路径的一处近似（见 ref/05）。
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
    // M5 的对照实验：故意把 b_g 初值置 0，其余不变。z 轴会从 0 爬到 0.0166，
    // 证明观测确实在"修正"零偏，而不只是让初值原样保持（主线只跑一遍看不出来）。
    if (bg_init_ == "zero") {
      RCLCPP_WARN(get_logger(),
                  "bg_init=zero：陀螺零偏初值取 0（对照实验）。静止段实测均值是 (%.6f, %.6f, "
                  "%.6f) rad/s，留着可以对比",
                  bg0[0], bg0[1], bg0[2]);
      for (int k = 0; k < 3; ++k) bg0[k] = 0.0;
    } else if (bg_init_ != "static") {
      RCLCPP_WARN(get_logger(), "bg_init=%s 不认识（只认 static / zero），按 static 处理",
                  bg_init_.c_str());
    }

    const eskf_imu::EZYX e0 = eskf_imu::euler_from_specific_force({fa[0], fa[1], fa[2]});
    const eskf_imu::Quat q0 = eskf_imu::quat_from_euler_zyx(e0.roll, e0.pitch, e0.yaw);

    // 静止段比力模长自检：1 附近 = g 单位（M1 的约定）；9.8 附近 = 已被换算过。
    const double fa_norm = std::sqrt(fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]);
    if (fa_norm > 1.5) {
      RCLCPP_WARN(get_logger(),
                  "静止段比力模长 = %.4f，不像 g 单位（应 ≈ 1）。/imu 是不是已经发 m/s² 了？"
                  "（imu_player_node 的 accel_to_mps2 应为 false），否则会重复换算",
                  fa_norm);
    }

    // b_a = |g|·(f̄ − R₀ᵀ·(0,0,1))。R₀ᵀ 的第三行就是 R₀ 的第三行。
    double ba0[3] = {0.0, 0.0, 0.0};
    if (estimate_position_) {
      if (acc_bias_init_ == "zero") {
        RCLCPP_WARN(get_logger(),
                    "acc_bias_init=zero：加计零偏初值取 0（对照实验，观察它能否被位置观测修正）");
      } else if (acc_bias_init_ != "static") {
        RCLCPP_WARN(get_logger(), "acc_bias_init=%s 不认识（只认 static / zero），按 static 处理",
                    acc_bias_init_.c_str());
      }
      if (acc_bias_init_ != "zero") {
        const eskf_imu::Mat3 R0 = eskf_imu::quat_to_rotation_matrix(q0);
        for (int k = 0; k < 3; ++k) ba0[k] = g_ms2_ * (fa[k] - R0(2, k));
      }
    }

    eskf_imu::Eskf::Noise noise;
    noise.sigma_g = sigma_g_;
    noise.sigma_bg = sigma_bg_;
    noise.sigma_a = sigma_a_;
    noise.sigma_ba = sigma_ba_;
    eskf_imu::Eskf::P0 p0;
    p0.theta = p0_theta_;
    p0.bg = p0_bg_;
    p0.p = p0_p_;
    p0.v = p0_v_;
    p0.ba = p0_ba_;
    eskf_.set_noise(noise);
    eskf_.set_gravity(g_ms2_);
    eskf_.set_r_min(r_min_);
    eskf_.init(q0, bg0, ba0, p0);
    for (int k = 0; k < 3; ++k) ba0_[k] = ba0[k];

    fout_.open(csv_out_);
    if (!fout_) {
      RCLCPP_FATAL(get_logger(), "无法打开输出文件: %s", csv_out_.c_str());
      throw std::runtime_error("csv_out open failed");
    }
    // 列名与单位（新列一律往后加，不改已有列名）：
    //   角度：度；四元数标量在前；P/S：只写对角线（姿态 rad²/rad、位置 m²/m）
    //   bg：rad/s；x,y,z：m；vx..vz：m/s；ba_x..ba_z：m/s²；upd/upd_p：本帧是否做了该次更新
    // 不开位置估计时**只写 M4 的那 26 列**，保证与 M4 的 ekf_out.csv 逐字节可比。
    fout_ << "time,dt,qw,qx,qy,qz,roll_deg,pitch_deg,yaw_deg,"
             "bg_x,bg_y,bg_z,p00,p11,p22,p33,p44,p55,"
             "z_x,z_y,z_z,nis,S00,S11,S22,upd";
    if (estimate_position_) {
      fout_ << ",x,y,z,vx,vy,vz,ba_x,ba_y,ba_z,"
               "zp_x,zp_y,zp_z,nis_p,Sp00,Sp11,Sp22,upd_p,"
               "Ppx,Ppy,Ppz,Pvx,Pvy,Pvz,Pbax,Pbay,Pbaz";
    }
    fout_ << "\n";

    for (size_t i = 0; i < n; ++i) {
      pending_.push_back(buffer_[i]);
    }
    buffer_.clear();
    started_ = true;

    RCLCPP_INFO(get_logger(),
                "初始化（前 %zu 帧）：b_g = (%.6f, %.6f, %.6f) rad/s，初始姿态 roll = %.3f° / "
                "pitch = %.3f°（参考 −0.197° / +0.546°）",
                n, bg0[0], bg0[1], bg0[2], e0.roll * 180.0 / M_PI, e0.pitch * 180.0 / M_PI);
    if (estimate_position_) {
      RCLCPP_INFO(get_logger(),
                  "静止段比力均值 = (%.6f, %.6f, %.6f) g，模长 %.6f（≈1 说明单位是 g）；"
                  "b_a 初值 = (%.6f, %.6f, %.6f) m/s² ≙ (%.6f, %.6f, %.6f) g",
                  fa[0], fa[1], fa[2], fa_norm, ba0[0], ba0[1], ba0[2], ba0[0] / g_ms2_,
                  ba0[1] / g_ms2_, ba0[2] / g_ms2_);
      RCLCPP_INFO(get_logger(),
                  "单位约定：加计读数 × %.5f → m/s²，重力向量取 (0, 0, +%.5f)（plan.md §2.5 的二选一）",
                  g_ms2_, g_ms2_);
    }
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
          for (int k = 0; k < 3; ++k) {
            obs.p[k] = po.p[k];
            obs.r[k] = po.r_diag[k];
            obs.r_pos[k] = po.r_pos_diag[k];
          }
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
      // 加计读数 g → m/s²（选做路径的 SI 约定）
      const double acc[3] = {s.acc[0] * g_ms2_, s.acc[1] * g_ms2_, s.acc[2] * g_ms2_};
      eskf_.predict(dt, s.gyro, acc);
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
    // 位置观测：与姿态观测同源同刻（都来自同一条 pose_cov 记录），先姿态后位置。
    // 两次更新互相独立（H 各选 3 列），顺序只影响一阶精度。
    double zp[3] = {0, 0, 0}, sdp[3] = {0, 0, 0}, nis_p = 0.0;
    bool pos_upd = false;
    if (estimate_position_ && use_position_update_ && obs.have) {
      const eskf_imu::UpdateResult r = eskf_.update_position(obs.p, obs.r_pos);
      for (int k = 0; k < 3; ++k) {
        zp[k] = r.z[k];
        sdp[k] = r.s_diag[k];
      }
      nis_p = r.nis;
      pos_upd = true;
      ++updated_pos_;
    }

    const eskf_imu::Quat& q = eskf_.q();
    const double* bg = eskf_.bg();
    const eskf_imu::Mat15& P = eskf_.P();
    for (int i = 0; i < 3; ++i) {
      bg_[i] = bg[i];
      // p00..p55 仍是姿态/零偏块的方差（写成 6 维时的下标顺序），M4 的脚本照旧可用
      last_p_[i] = P(eskf_imu::Eskf::kTheta + i, eskf_imu::Eskf::kTheta + i);
      last_p_[3 + i] = P(eskf_imu::Eskf::kBg + i, eskf_imu::Eskf::kBg + i);
    }
    if (estimate_position_) {
      const double* p = eskf_.p();
      const double* v = eskf_.v();
      const double* ba = eskf_.ba();
      for (int i = 0; i < 3; ++i) {
        p_[i] = p[i];
        v_[i] = v[i];
        ba_[i] = ba[i];
        last_ppv_[i] = P(eskf_imu::Eskf::kP + i, eskf_imu::Eskf::kP + i);
        last_ppv_[3 + i] = P(eskf_imu::Eskf::kV + i, eskf_imu::Eskf::kV + i);
        last_ppv_[6 + i] = P(eskf_imu::Eskf::kBa + i, eskf_imu::Eskf::kBa + i);
      }
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
          << sd[2] << "," << (obs.have ? 1 : 0);
    if (estimate_position_) {
      // 位置用 fixed 6（m 量级，1e-6 m 足够）；残差/NIS 与 S 用科学计数法
      fout_ << std::fixed << std::setprecision(6) << "," << p_[0] << "," << p_[1] << "," << p_[2]
            << "," << v_[0] << "," << v_[1] << "," << v_[2];
      fout_ << std::scientific << std::setprecision(6) << "," << ba_[0] << "," << ba_[1] << ","
            << ba_[2];
      fout_ << "," << zp[0] << "," << zp[1] << "," << zp[2];
      fout_ << std::fixed << std::setprecision(6) << "," << nis_p;
      fout_ << std::scientific << std::setprecision(6) << "," << sdp[0] << "," << sdp[1] << ","
            << sdp[2] << "," << (pos_upd ? 1 : 0);
      for (int i = 0; i < 9; ++i) fout_ << "," << last_ppv_[i];
    }
    fout_ << "\n";
    ++written_;
  }

  std::string csv_out_;
  int bias_window_frames_ = 1000;
  std::string bg_init_ = "static";
  double sigma_g_ = 0.0032, sigma_bg_ = 1e-4;
  double p0_theta_ = 1e-3, p0_bg_ = 1e-4;
  bool use_pose_update_ = true;
  double r_min_ = 1e-8;
  int max_pending_frames_ = 3000;
  bool estimate_position_ = false;
  bool use_position_update_ = true;
  double g_ms2_ = 9.80665;
  std::string acc_bias_init_ = "static";
  double p0_p_ = 1e-2, p0_v_ = 1e-2, p0_ba_ = 5e-2;
  double sigma_a_ = 0.12, sigma_ba_ = 5e-4;
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
  double ba_[3] = {0, 0, 0};
  double ba0_[3] = {0, 0, 0};
  double p_[3] = {0, 0, 0};
  double v_[3] = {0, 0, 0};
  double last_p_[6] = {0, 0, 0, 0, 0, 0};
  double last_ppv_[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::ofstream fout_;
  size_t written_ = 0, skipped_dt_ = 0, updated_ = 0, no_obs_ = 0;
  size_t stale_drained_ = 0, flushed_ = 0, pose_msgs_ = 0, zero_cov_pose_ = 0;
  size_t updated_pos_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EskfNode>();
  rclcpp::spin(node);
  node->finish();
  rclcpp::shutdown();
  return 0;
}
