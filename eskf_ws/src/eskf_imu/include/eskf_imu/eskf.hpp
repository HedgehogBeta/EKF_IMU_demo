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

// 3x3 求逆（S = HPHᵀ + R 对称正定），伴随矩阵除以行列式。
// 行列式退化时返回零矩阵：卡尔曼增益随之归零，等价于这一步不做修正，比发散安全。
inline Mat3 mat3_inverse(const Mat3& a) {
  const double det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
                     a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
                     a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
  Mat3 inv;
  if (std::abs(det) < 1e-300) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) inv(i, j) = 0.0;
    }
    return inv;
  }
  const double id = 1.0 / det;
  inv(0, 0) = (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * id;
  inv(0, 1) = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * id;
  inv(0, 2) = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * id;
  inv(1, 0) = (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * id;
  inv(1, 1) = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * id;
  inv(1, 2) = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * id;
  inv(2, 0) = (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * id;
  inv(2, 1) = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * id;
  inv(2, 2) = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * id;
  return inv;
}

// 一次观测更新的中间量，写进 CSV 供 M5 做一致性检验
struct UpdateResult {
  double z[3] = {0.0, 0.0, 0.0};       // 残差 2·vec(q̂⁻¹⊗q_obs)，rad
  double s_diag[3] = {0.0, 0.0, 0.0};  // S = HPHᵀ + R 的对角线
  double nis = 0.0;                    // zᵀS⁻¹z，3 维观测下均值应接近 3
};

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

  // 观测更新：直接观测姿态本身（h(q) = q），H = [I₃ 0₃ₓ₃]。
  // q_obs 为已对齐到当前时刻的姿态观测，r_diag = (cov_33, cov_44, cov_55) 逐帧读取。
  UpdateResult update(const Quat& q_obs, const double r_diag[3]) {
    UpdateResult out;

    // ① 符号对齐：q 与 -q 是同一旋转。不对齐则残差方向整体反过来，修正会推错方向。
    Quat qo = q_obs;
    if (qdot(qo, q_) < 0.0) qo = {-qo.w, -qo.x, -qo.y, -qo.z};

    // ② 残差用旋转向量差，不用四元数分量差：z = 2·vec(q̂⁻¹ ⊗ q_obs) ∈ ℝ³
    const Quat dq = qmul(qconj(q_), qo);
    out.z[0] = 2.0 * dq.x;
    out.z[1] = 2.0 * dq.y;
    out.z[2] = 2.0 * dq.z;

    // ③ S = HPHᵀ + R = P_θθ + R。R 给下限兜底，避免全零协方差让 S 病态。
    Mat3 S;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double r = (r_diag[i] > r_min_) ? r_diag[i] : r_min_;
        S(i, j) = P_(i, j) + ((i == j) ? r : 0.0);
      }
    }
    const Mat3 Sinv = mat3_inverse(S);

    // ④ K = PHᵀS⁻¹（6×3），δx = K·z。H 只取前 3 列，所以 K 就是 P 的左块乘 S⁻¹。
    double K[6][3];
    for (int i = 0; i < 6; ++i) {
      for (int k = 0; k < 3; ++k) {
        double acc = 0.0;
        for (int j = 0; j < 3; ++j) acc += P_(i, j) * Sinv(j, k);
        K[i][k] = acc;
      }
    }
    double dx[6] = {};
    for (int i = 0; i < 6; ++i) {
      for (int k = 0; k < 3; ++k) dx[i] += K[i][k] * out.z[k];
    }
    for (int i = 0; i < 3; ++i) {
      out.s_diag[i] = S(i, i);
      double acc = 0.0;
      for (int k = 0; k < 3; ++k) acc += out.z[k] * Sinv(k, i);
      out.nis += acc * out.z[i];
    }

    // ⑤ 注入 + 归一化。δx 是局部量，用完即弃 —— 误差状态"清零"由此天然满足，
    //    不会出现同一份误差被重复注入（plan.md 陷阱 10）。
    q_ = qnormalize(qmul(q_, qexp({dx[0], dx[1], dx[2]})));
    for (int i = 0; i < 3; ++i) bg_[i] += dx[3 + i];

    // ⑥ P ← (I − KH)P = P − K·P_θₗ 行（KH 只留 H 的前 3 列）。
    //    必须写进临时矩阵：右式的 P_(k,j) 要读更新前的 P，逐行原地减会读到已改过的行。
    Mat6 Pn;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        double acc = 0.0;
        for (int k = 0; k < 3; ++k) acc += K[i][k] * P_(k, j);
        Pn(i, j) = P_(i, j) - acc;
      }
    }
    P_ = Pn;
    return out;
  }

  const Quat& q() const { return q_; }
  const double* bg() const { return bg_; }
  const Mat6& P() const { return P_; }

  void set_noise(double sigma_g, double sigma_bg) {
    sigma_g_ = sigma_g;
    sigma_bg_ = sigma_bg;
  }

  // R 对角线的下限（rad²）。题目规定 R 逐帧取自 pose_cov.csv，本身不可调；
  // 这个下限只是数值兜底，防止 0 值让 S 病态（前 20 帧协方差全零）。
  void set_r_min(double r_min) { r_min_ = r_min; }

 private:
  Quat q_;
  double bg_[3] = {0.0, 0.0, 0.0};
  Mat6 P_;
  double sigma_g_ = 0.0032;   // 陀螺白噪声 std，rad/s（静止段统计）
  double sigma_bg_ = 1e-4;    // 零偏随机游走 std，rad/s/√s
  double r_min_ = 1e-8;       // R 对角线下限
};

}  // namespace eskf_imu
