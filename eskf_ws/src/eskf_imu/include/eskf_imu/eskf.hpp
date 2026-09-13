#pragma once

#include <cmath>

#include "eskf_imu/quaternion.hpp"

namespace eskf_imu {

struct Mat6 {
  double m[6][6] = {};  // 默认全零

  double& operator()(int r, int c) { return m[r][c]; }
  double operator()(int r, int c) const { return m[r][c]; }

  static Mat6 identity() {
    Mat6 I;
    for (int i = 0; i < 6; ++i) I(i, i) = 1.0;
    return I;
  }
};

inline Mat6 mat6_mul(const Mat6& a, const Mat6& b) {
  Mat6 r;
  for (int i = 0; i < 6; ++i) {
    for (int k = 0; k < 6; ++k) {
      const double aik = a(i, k);
      if (aik == 0.0) continue;
      for (int j = 0; j < 6; ++j) r(i, j) += aik * b(k, j);
    }
  }
  return r;
}

// a * bᵀ（b 转置后相乘），用于 P ← F·P·Fᵀ
inline Mat6 mat6_mul_by_transpose(const Mat6& a, const Mat6& b) {
  Mat6 r;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      double s = 0.0;
      for (int k = 0; k < 6; ++k) s += a(i, k) * b(j, k);
      r(i, j) = s;
    }
  }
  return r;
}

class Eskf {
 public:
  // p0_theta / p0_bg：初始姿态 / 零偏不确定度（标准差），P0 = diag(σ²I₃, σ²I₃)。
  void init(const Quat& q0, const double bg0[3], double p0_theta, double p0_bg) {
    q_ = qnormalize(q0);
    for (int i = 0; i < 3; ++i) bg_[i] = bg0[i];
    P_ = Mat6();
    for (int i = 0; i < 3; ++i) {
      P_(i, i) = p0_theta * p0_theta;
      P_(3 + i, 3 + i) = p0_bg * p0_bg;
    }
  }

  // gyro 为陀螺原始测量（rad/s）。dt 取时间戳差分。
  void predict(double dt, const double gyro[3]) {
    if (dt <= 0.0) return;  // 时间戳回退/重复

    const double w[3] = {gyro[0] - bg_[0], gyro[1] - bg_[1], gyro[2] - bg_[2]};

    // 标称状态推进：q ← q ⊗ Exp(ω·Δt)，b_g 不变（零偏由观测步估计）
    q_ = qnormalize(qmul(q_, qexp({w[0] * dt, w[1] * dt, w[2] * dt})));

    // F = [[I − [ω×]Δt, −Δt·I], [0, I]]
    // 左上块：姿态误差被当前旋转"搬运"；右上块 −Δt·I：零偏误差随时间积分成姿态误差。
    Mat6 F = Mat6::identity();
    F(0, 1) = w[2] * dt;
    F(0, 2) = -w[1] * dt;
    F(1, 0) = -w[2] * dt;
    F(1, 2) = w[0] * dt;
    F(2, 0) = w[1] * dt;
    F(2, 1) = -w[0] * dt;
    for (int i = 0; i < 3; ++i) F(i, 3 + i) = -dt;

    // Q = diag(σg²·Δt²·I₃, σbg²·Δt·I₃)
    // 上块：陀螺白噪声在本步积分中引入的姿态误差（随 Δt² 增长）；
    // 下块：零偏随机游走（题面明确要求，没有它零偏永远学不到）。
    Mat6 Qc;
    for (int i = 0; i < 3; ++i) {
      Qc(i, i) = sigma_g_ * sigma_g_ * dt * dt;
      Qc(3 + i, 3 + i) = sigma_bg_ * sigma_bg_ * dt;
    }

    P_ = mat6_mul_by_transpose(mat6_mul(F, P_), F);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) P_(i, j) += Qc(i, j);
    }
  }

  const Quat& q() const { return q_; }
  const double* bg() const { return bg_; }
  const Mat6& P() const { return P_; }

  void set_noise(double sigma_g, double sigma_bg) {
    sigma_g_ = sigma_g;
    sigma_bg_ = sigma_bg;
  }

 private:
  Quat q_;
  double bg_[3] = {0.0, 0.0, 0.0};
  Mat6 P_;
  double sigma_g_ = 0.0032;   // 陀螺白噪声 std，rad/s（静止段统计）
  double sigma_bg_ = 1e-4;    // 零偏随机游走 std，rad/s/√s
};

}  // namespace eskf_imu
