#pragma once

#include <cstddef>
#include <deque>

#include "eskf_imu/quaternion.hpp"

namespace eskf_imu {

// 对齐到某个 IMU 时刻的一次姿态观测。
// valid=false 表示协方差整行为 0（观测尚未初始化，pose_cov.csv 开头 20 帧），
// 调用方应当跳过这次更新而不是把 R 当成 0 用。
struct PoseObs {
  Quat q;
  double r_diag[3] = {0.0, 0.0, 0.0};
  bool valid = false;
};

// 位姿观测的时间对齐。不依赖 ROS，可离线喂 CSV 做单测。
//
// 两件事：
//  1) 入缓冲时**符号展开**：与上一条样本点积为负就整体取反。姿态序列里混有 q -> -q
//     的毛刺（实测 pose_cov.csv 第 17957 帧，真转角只有 0.45°），不展开则相邻两样本
//     之间的 SLERP 会走 359.55° 长弧、插值中点偏 180°，残差瞬间跳到 ≈2。
//  2) 取值时找**括号**（前一条 <= t <= 后一条）做 SLERP，协方差取最近邻
//     （CONTEXT.md「插值对齐」：姿态插值、协方差最近邻）。
//
// 调用方必须按 t 不减的顺序调用 lookup（IMU 帧天然如此）。样本不够时返回 kNeedMore
// 表示"等等后面的样本"；kNoObsYet 表示 t 早于最早的样本，这一帧永远等不到括号。
class PoseAligner {
 public:
  enum Status { kOk, kNeedMore, kNoObsYet };

  // 原始观测入缓冲。q_raw 不做预处理，符号展开在这里做。
  // 时间戳必须递增，否则该样本被丢弃并计数（回放数据是递增的，出现即异常）。
  void push(double t, const Quat& q_raw, double r00, double r11, double r22, bool valid) {
    Quat q = q_raw;
    if (has_last_ && qdot(q, last_q_) < 0.0) {
      q = {-q.w, -q.x, -q.y, -q.z};  // 展开到与上一条同半球，SLERP 才会走短弧
    }
    last_q_ = q;
    has_last_ = true;

    if (!buf_.empty() && t <= buf_.back().t) {
      ++out_of_order_;
      return;
    }
    buf_.push_back(Sample{t, q, {r00, r11, r22}, valid});
  }

  // 取时刻 t 的观测。kOk 时填好 out。
  Status lookup(double t, PoseObs& out) {
    if (buf_.empty()) return kNeedMore;
    if (t < buf_.front().t) return kNoObsYet;  // 样本按 t 递增，前面不会再来更早的

    // 推进 cursor_ 到"最后一条 stamp <= t"的样本
    while (cursor_ + 1 < buf_.size() && buf_[cursor_ + 1].t <= t) ++cursor_;

    // 时间戳正好落在某条样本上：这条就是 t 时刻的观测，不需要右括号
    if (buf_[cursor_].t == t) {
      out.q = buf_[cursor_].q;
      fill(out, buf_[cursor_]);
      consume();
      return kOk;
    }
    if (cursor_ + 1 >= buf_.size()) return kNeedMore;  // 最新的样本都在 t 之前，等未来样本

    const Sample& a = buf_[cursor_];
    const Sample& b = buf_[cursor_ + 1];
    const double span = b.t - a.t;
    const double u = span > 0.0 ? (t - a.t) / span : 0.0;
    out.q = qslerp(a.q, b.q, u);
    fill(out, (u < 0.5) ? a : b);  // 只取协方差与有效性，姿态用插值结果
    consume();
    return kOk;
  }

  size_t size() const { return buf_.size(); }
  size_t out_of_order() const { return out_of_order_; }

 private:
  struct Sample {
    double t;
    Quat q;
    double r_diag[3];
    bool valid;
  };

  // 只搬协方差与有效性；姿态一定是插值结果，不能在这里被样本覆盖掉
  static void fill(PoseObs& out, const Sample& s) {
    out.r_diag[0] = s.r_diag[0];
    out.r_diag[1] = s.r_diag[1];
    out.r_diag[2] = s.r_diag[2];
    out.valid = s.valid;
  }

  // 丢掉已消费的前缀，但保留 cursor_ 那条当下一帧的左括号
  void consume() {
    buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(cursor_));
    cursor_ = 0;
  }

  std::deque<Sample> buf_;
  size_t cursor_ = 0;
  Quat last_q_;          // 最后一条入缓冲的（已展开）样本，供下一条做符号展开
  bool has_last_ = false;
  size_t out_of_order_ = 0;
};

}  // namespace eskf_imu
