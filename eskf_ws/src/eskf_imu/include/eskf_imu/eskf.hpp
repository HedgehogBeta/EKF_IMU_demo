#pragma once

#include <cmath>

#include "eskf_imu/quaternion.hpp"

namespace eskf_imu {

// ---------------- 手写方阵 ----------------
// N×N 行主序方阵。不用 Eigen：算法层只用标准库，能脱离 ROS 单独编译。
template <int N>
struct MatN {
  double m[N][N] = {};  // 默认全零

  double& operator()(int r, int c) { return m[r][c]; }
  double operator()(int r, int c) const { return m[r][c]; }

  static MatN identity() {
    MatN I;
    for (int i = 0; i < N; ++i) I(i, i) = 1.0;
    return I;
  }
};

// a * b
template <int N>
inline MatN<N> mat_mul(const MatN<N>& a, const MatN<N>& b) {
  MatN<N> r;
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < N; ++k) {
      const double aik = a(i, k);
      if (aik == 0.0) continue;  // 零块直接跳过（15 维 F 是稀疏块状，省掉大半乘法）
      for (int j = 0; j < N; ++j) r(i, j) += aik * b(k, j);
    }
  }
  return r;
}

// a * bᵀ（b 转置后相乘），用于 P ← F·P·Fᵀ
template <int N>
inline MatN<N> mat_mul_by_transpose(const MatN<N>& a, const MatN<N>& b) {
  MatN<N> r;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < N; ++k) s += a(i, k) * b(j, k);
      r(i, j) = s;
    }
  }
  return r;
}

using Mat15 = MatN<15>;

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

// 反对称矩阵 [v]×：满足 [v]× u = v × u
inline Mat3 skew(const Vec3& v) {
  Mat3 s;
  s(0, 0) = 0.0;
  s(0, 1) = -v.z;
  s(0, 2) = v.y;
  s(1, 0) = v.z;
  s(1, 1) = 0.0;
  s(1, 2) = -v.x;
  s(2, 0) = -v.y;
  s(2, 1) = v.x;
  s(2, 2) = 0.0;
  return s;
}

inline Vec3 mat3_mul_vec(const Mat3& a, const Vec3& v) {
  Vec3 r;
  r.x = a(0, 0) * v.x + a(0, 1) * v.y + a(0, 2) * v.z;
  r.y = a(1, 0) * v.x + a(1, 1) * v.y + a(1, 2) * v.z;
  r.z = a(2, 0) * v.x + a(2, 1) * v.y + a(2, 2) * v.z;
  return r;
}

// 一次观测更新的中间量，写进 CSV 供评估脚本做一致性检验
struct UpdateResult {
  double z[3] = {0.0, 0.0, 0.0};       // 残差（姿态：2·vec(q̂⁻¹⊗q_obs) rad；位置：p_obs − p̂ m）
  double s_diag[3] = {0.0, 0.0, 0.0};  // S = HPHᵀ + R 的对角线
  double nis = 0.0;                    // zᵀS⁻¹z，3 维观测下均值应接近 3
};

// 误差状态卡尔曼滤波（ESKF）。状态分块固定为 15 维：
//
//   标称状态：p(3) v(3) q(4→3) b_g(3) b_a(3)
//   误差状态：δx = [δp, δv, δθ, δb_g, δb_a] ∈ ℝ¹⁵，P 为 15×15
// 单位：位置 m、速度 m/s、陀螺 rad/s、**加速度 m/s²、加计零偏 m/s²**。
// 实测 IMU 数据里加速度是 g，节点负责换算（见 eskf_node 的 g_ms2 参数）。
// 坐标约定：q 表示机体系 → 世界系（R_WB），体轴角增量右乘。
class Eskf {
 public:
  static constexpr int kDim = 15;
  // 误差状态各分块的起点下标
  static constexpr int kP = 0;      // δp
  static constexpr int kV = 3;      // δv
  static constexpr int kTheta = 6;  // δθ
  static constexpr int kBg = 9;     // δb_g
  static constexpr int kBa = 12;    // δb_a

  // P₀ = diag(σ² I₃, …)，每一段一个标准差
  struct P0 {
    double theta = 1e-3;  // rad：比力对准的误差量级
    double bg = 1e-4;     // rad/s：静止段均值已经估得比较准
    double p = 1e-2;      // m：观测原点与滤波起点的差异（且起点前 0.17 s 静止）
    double v = 1e-2;      // m/s：静止段实测 |v| ≈ 0.02 m/s
    double ba = 5e-2;     // m/s²：静止段估计被尺度因子污染（实测差 0.57%，约 0.057 m/s²）
  };

  // Q 的四个噪声源（题面要求含零偏随机游走）
  struct Noise {
    double sigma_g = 0.0032;   // 陀螺白噪声 std，rad/s（静止段统计）
    double sigma_bg = 1e-4;    // 陀螺零偏随机游走，rad/s/√s
    double sigma_a = 0.12;     // 加计白噪声 std，m/s²（静止段 x 轴 0.0123 g）
    double sigma_ba = 5e-4;    // 加计零偏随机游走，m/s²/√s
  };

  // p0 / v0 取零：世界系原点即 FAST-LIO 的起点（实测首帧观测 x=y=z=0），
  // 且机器人从静止起步（v=0 是已知信息，和 b_g 用静止段均值作初值同一个道理）。
  void init(const Quat& q0, const double bg0[3], const double ba0[3], const P0& p0) {
    q_ = qnormalize(q0);
    for (int i = 0; i < 3; ++i) {
      p_[i] = 0.0;
      v_[i] = 0.0;
      bg_[i] = bg0[i];
      ba_[i] = ba0[i];
    }
    P_ = Mat15();
    for (int i = 0; i < 3; ++i) {
      P_(kP + i, kP + i) = p0.p * p0.p;
      P_(kV + i, kV + i) = p0.v * p0.v;
      P_(kTheta + i, kTheta + i) = p0.theta * p0.theta;
      P_(kBg + i, kBg + i) = p0.bg * p0.bg;
      P_(kBa + i, kBa + i) = p0.ba * p0.ba;
    }
  }

  // gyro: rad/s；acc: m/s²（已换算，见类注释）。dt 取时间戳差分。
  void predict(double dt, const double gyro[3], const double acc[3]) {
    if (dt <= 0.0) return;  // 时间戳回退/重复

    const double w[3] = {gyro[0] - bg_[0], gyro[1] - bg_[1], gyro[2] - bg_[2]};

    // ① 标称状态推进
    //    姿态：q ← q ⊗ Exp(ω·Δt)
    q_ = qnormalize(qmul(q_, qexp({w[0] * dt, w[1] * dt, w[2] * dt})));
    //    位置/速度：ã = a_m − b_a（去零偏）→ 旋转到世界系 → 减去重力
    //              a_w = R·ã − g_W，v ← v + a_w·Δt，p ← p + v·Δt
    const Mat3 R = quat_to_rotation_matrix(q_);
    const Vec3 f = {acc[0] - ba_[0], acc[1] - ba_[1], acc[2] - ba_[2]};
    const Vec3 aw = mat3_mul_vec(R, f);
    const double awv[3] = {aw.x, aw.y, aw.z};
    for (int i = 0; i < 3; ++i) {
      p_[i] += v_[i] * dt;
      v_[i] += (awv[i] - g_w_[i]) * dt;
    }

    // ② F = I + A_c·Δt（一阶欧拉离散）。A_c 的块结构：
    //      δp' = δv
    //      δv' = −R[f]× δθ − R δb_a
    //      δθ' = −[ω]× δθ − δb_g
    //      δb_g' = 0,  δb_a' = 0
    Mat15 F = Mat15::identity();
    const Mat3 Sf = skew(f);
    for (int i = 0; i < 3; ++i) {
      F(kP + i, kV + i) = dt;
      F(kTheta + i, kBg + i) = -dt;
      for (int j = 0; j < 3; ++j) {
        double rf = 0.0;
        for (int k = 0; k < 3; ++k) rf += R(i, k) * Sf(k, j);
        F(kV + i, kTheta + j) = -rf * dt;
        F(kV + i, kBa + j) = -R(i, j) * dt;
      }
    }
    F(kTheta + 0, kTheta + 1) = w[2] * dt;
    F(kTheta + 0, kTheta + 2) = -w[1] * dt;
    F(kTheta + 1, kTheta + 0) = -w[2] * dt;
    F(kTheta + 1, kTheta + 2) = w[0] * dt;
    F(kTheta + 2, kTheta + 0) = w[1] * dt;
    F(kTheta + 2, kTheta + 1) = -w[0] * dt;

    // ③ Q：四个噪声源各一块
    //    δv 块 σ_a²Δt²、δθ 块 σ_g²Δt²：白噪声在这一步里积成速度/姿态误差
    //    δb_g / δb_a 块 σ²Δt：零偏随机游走（题面要求 Q 含零偏项）
    Mat15 Qc;
    for (int i = 0; i < 3; ++i) {
      Qc(kV + i, kV + i) = noise_.sigma_a * noise_.sigma_a * dt * dt;
      Qc(kTheta + i, kTheta + i) = noise_.sigma_g * noise_.sigma_g * dt * dt;
      Qc(kBg + i, kBg + i) = noise_.sigma_bg * noise_.sigma_bg * dt;
      Qc(kBa + i, kBa + i) = noise_.sigma_ba * noise_.sigma_ba * dt;
    }

    // ④ P ← F·P·Fᵀ + Q
    P_ = mat_mul_by_transpose(mat_mul(F, P_), F);
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < kDim; ++j) P_(i, j) += Qc(i, j);
    }
  }

  // 姿态观测：直接观测姿态本身（h(q) = q），H = [0 0 I₃ 0 0]（选 δθ 那 3 列）。
  // q_obs 为已对齐到当前时刻的姿态观测，r_diag = (cov_33, cov_44, cov_55) 逐帧读取。
  UpdateResult update(const Quat& q_obs, const double r_diag[3]) {
    // 符号对齐：q 与 -q 是同一旋转。不对齐则残差方向整体反过来，修正会推错方向。
    Quat qo = q_obs;
    if (qdot(qo, q_) < 0.0) qo = {-qo.w, -qo.x, -qo.y, -qo.z};

    // 残差用旋转向量差，不用四元数分量差：z = 2·vec(q̂⁻¹ ⊗ q_obs) ∈ ℝ³
    const Quat dq = qmul(qconj(q_), qo);
    const double z[3] = {2.0 * dq.x, 2.0 * dq.y, 2.0 * dq.z};
    static const int kCol[3] = {kTheta, kTheta + 1, kTheta + 2};
    return apply_observation(kCol, z, r_diag);
  }

  // 位置观测：观测值 pose_cov 的 x,y,z（m），H = [I₃ 0 …]（选 δp 那 3 列）。
  // r_diag = (cov_00, cov_11, cov_22) 逐帧读取。
  // 残差就是向量差 z = p_obs − p̂（位置是线性量，没有姿态那种符号问题）。
  UpdateResult update_position(const double p_obs[3], const double r_diag[3]) {
    double z[3];
    for (int i = 0; i < 3; ++i) z[i] = p_obs[i] - p_[i];
    static const int kCol[3] = {kP, kP + 1, kP + 2};
    return apply_observation(kCol, z, r_diag);
  }

  const Quat& q() const { return q_; }
  const double* bg() const { return bg_; }
  const double* p() const { return p_; }
  const double* v() const { return v_; }
  const double* ba() const { return ba_; }
  const Mat15& P() const { return P_; }

  void set_noise(const Noise& n) { noise_ = n; }
  void set_gravity(double g_ms2) {
    g_w_[0] = 0.0;
    g_w_[1] = 0.0;
    g_w_[2] = g_ms2;  // 世界系 z 朝上，比力参考取 (0,0,+|g|)（静止时加计读数是 +1 g）
  }

  // R 对角线的下限。题目规定 R 逐帧取自 pose_cov.csv，本身不可调；
  // 这个下限只是数值兜底，防止 0 值让 S 病态（前 20 帧协方差全零）。
  // 单位随观测：姿态 rad²、位置 m²。
  void set_r_min(double r_min) { r_min_ = r_min; }

 private:
  // 通用观测更新：H 只选 3 列（col[0..2]），其余列为 0。
  //   S = P_cc + R，K = P[:,c] S⁻¹（15×3），δx = K z，
  //   注入 + 误差状态清零（δx 是局部量，用完即弃），P ← (I − KH)P。
  UpdateResult apply_observation(const int col[3], const double z[3], const double r_diag[3]) {
    UpdateResult out;
    for (int i = 0; i < 3; ++i) out.z[i] = z[i];

    // ③ S = HPHᵀ + R。R 给下限兜底，避免全零协方差让 S 病态。
    Mat3 S;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double r = (r_diag[i] > r_min_) ? r_diag[i] : r_min_;
        S(i, j) = P_(col[i], col[j]) + ((i == j) ? r : 0.0);
      }
    }
    const Mat3 Sinv = mat3_inverse(S);

    // ④ K = PHᵀS⁻¹（15×3），δx = K·z
    double K[kDim][3];
    for (int i = 0; i < kDim; ++i) {
      for (int k = 0; k < 3; ++k) {
        double acc = 0.0;
        for (int j = 0; j < 3; ++j) acc += P_(i, col[j]) * Sinv(j, k);
        K[i][k] = acc;
      }
    }
    double dx[kDim] = {};
    for (int i = 0; i < kDim; ++i) {
      for (int k = 0; k < 3; ++k) dx[i] += K[i][k] * z[k];
    }
    for (int i = 0; i < 3; ++i) {
      out.s_diag[i] = S(i, i);
      double acc = 0.0;
      for (int k = 0; k < 3; ++k) acc += z[k] * Sinv(k, i);
      out.nis += acc * z[i];
    }

    // ⑤ 注入 + 归一化
    for (int i = 0; i < 3; ++i) {
      p_[i] += dx[kP + i];
      v_[i] += dx[kV + i];
    }
    q_ = qnormalize(qmul(q_, qexp({dx[kTheta], dx[kTheta + 1], dx[kTheta + 2]})));
    for (int i = 0; i < 3; ++i) {
      bg_[i] += dx[kBg + i];
      ba_[i] += dx[kBa + i];
    }

    // ⑥ P ← (I − KH)P = P − K·P_c 行（KH 只留 H 的那 3 列对应的行）。
    //    必须写进临时矩阵：右式的 P_(col[k], j) 要读更新前的 P，原地逐行减会读到已改过的行。
    Mat15 Pn;
    for (int i = 0; i < kDim; ++i) {
      for (int j = 0; j < kDim; ++j) {
        double acc = 0.0;
        for (int k = 0; k < 3; ++k) acc += K[i][k] * P_(col[k], j);
        Pn(i, j) = P_(i, j) - acc;
      }
    }
    P_ = Pn;
    return out;
  }

  Quat q_;
  double p_[3] = {0.0, 0.0, 0.0};   // m
  double v_[3] = {0.0, 0.0, 0.0};   // m/s
  double bg_[3] = {0.0, 0.0, 0.0};  // rad/s
  double ba_[3] = {0.0, 0.0, 0.0};  // m/s²
  Mat15 P_;
  Noise noise_;
  double g_w_[3] = {0.0, 0.0, 9.80665};
  double r_min_ = 1e-8;
};

}  // namespace eskf_imu
