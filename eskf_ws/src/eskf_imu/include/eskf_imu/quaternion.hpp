#pragma once

#include <cmath>

namespace eskf_imu {

// 四元数运算用 test_quaternion 与 Eigen 对照。
// 约定：q 表示机体系 → 世界系的旋转 R_WB；体轴角增量右乘，q <- q ⊗ Exp(ω Δt)。
// 分量顺序：w 在前（x,y,z 为向量部分）。
struct Quat {
  double w = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

// 3x3 旋转矩阵，行主序，v_world = R * v_body
struct Mat3 {
  double m[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  double& operator()(int r, int c) { return m[r][c]; }
  double operator()(int r, int c) const { return m[r][c]; }
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

inline Quat qmul(const Quat& a, const Quat& b) {
  Quat r;
  r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
  r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
  r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
  r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
  return r;
}

inline Quat qconj(const Quat& q) { return {q.w, -q.x, -q.y, -q.z}; }

inline double qdot(const Quat& a, const Quat& b) {
  return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double qnorm(const Quat& q) { return std::sqrt(qdot(q, q)); }

inline Quat qnormalize(const Quat& q) {
  const double n = qnorm(q);
  if (n < 1e-15) return q;
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}

// 指数映射：Exp(phi) = [cos(|phi|/2), phi_hat * sin(|phi|/2)]。
// |phi| -> 0 时退化为 [1, phi/2]，与极限一致。
inline Quat qexp(const Vec3& phi) {
  const double angle = std::sqrt(phi.x * phi.x + phi.y * phi.y + phi.z * phi.z);
  if (angle < 1e-12) {
    return {1.0, 0.5 * phi.x, 0.5 * phi.y, 0.5 * phi.z};
  }
  const double s = std::sin(0.5 * angle) / angle;
  return {std::cos(0.5 * angle), s * phi.x, s * phi.y, s * phi.z};
}

inline Mat3 quat_to_rotation_matrix(const Quat& q) {
  Mat3 R;
  const double ww = q.w * q.w, xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
  const double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
  const double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
  R(0, 0) = ww + xx - yy - zz;
  R(0, 1) = 2.0 * (xy - wz);
  R(0, 2) = 2.0 * (xz + wy);
  R(1, 0) = 2.0 * (xy + wz);
  R(1, 1) = ww - xx + yy - zz;
  R(1, 2) = 2.0 * (yz - wx);
  R(2, 0) = 2.0 * (xz - wy);
  R(2, 1) = 2.0 * (yz + wx);
  R(2, 2) = ww - xx - yy + zz;
  return R;
}

// 旋转矩阵 -> 四元数（Shepperd 分支法：按迹/主对角元选数值最稳的分支）
inline Quat quat_from_rotation_matrix(const Mat3& R) {
  const double trace = R(0, 0) + R(1, 1) + R(2, 2);
  Quat q;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;  // s = 4w
    q.w = 0.25 * s;
    q.x = (R(2, 1) - R(1, 2)) / s;
    q.y = (R(0, 2) - R(2, 0)) / s;
    q.z = (R(1, 0) - R(0, 1)) / s;
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    const double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;  // s = 4x
    q.w = (R(2, 1) - R(1, 2)) / s;
    q.x = 0.25 * s;
    q.y = (R(0, 1) + R(1, 0)) / s;
    q.z = (R(0, 2) + R(2, 0)) / s;
  } else if (R(1, 1) > R(2, 2)) {
    const double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;  // s = 4y
    q.w = (R(0, 2) - R(2, 0)) / s;
    q.x = (R(0, 1) + R(1, 0)) / s;
    q.y = 0.25 * s;
    q.z = (R(1, 2) + R(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;  // s = 4z
    q.w = (R(1, 0) - R(0, 1)) / s;
    q.x = (R(0, 2) + R(2, 0)) / s;
    q.y = (R(1, 2) + R(2, 1)) / s;
    q.z = 0.25 * s;
  }
  return qnormalize(q);
}

// ZYX 内旋欧拉角（yaw 绕 z -> pitch 绕 y -> roll 绕 x），返回 (roll, pitch, yaw)
struct EZYX {
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

inline EZYX quat_to_euler_zyx(const Quat& q) {
  const Mat3 R = quat_to_rotation_matrix(q);
  EZYX e;
  e.roll = std::atan2(R(2, 1), R(2, 2));
  e.pitch = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
  e.yaw = std::atan2(R(1, 0), R(0, 0));
  return e;
}

// 由比力（静止时 = 重力在机体系的投影，单位 g）解算初始 roll/pitch，yaw 不可观设 0。
// 静止时 f_b = R_WB^T * (0,0,1)（g 单位），推出 roll = atan2(fy, fz)，
// pitch = atan2(-fx, sqrt(fy^2 + fz^2))。
inline EZYX euler_from_specific_force(const Vec3& f) {
  EZYX e;
  e.roll = std::atan2(f.y, f.z);
  e.pitch = std::atan2(-f.x, std::sqrt(f.y * f.y + f.z * f.z));
  e.yaw = 0.0;
  return e;
}

// 欧拉角 -> 四元数（与 quat_to_euler_zyx 互逆），用于给初始对准的结果构造 q0
inline Quat quat_from_euler_zyx(double roll, double pitch, double yaw) {
  const double cr = std::cos(0.5 * roll), sr = std::sin(0.5 * roll);
  const double cp = std::cos(0.5 * pitch), sp = std::sin(0.5 * pitch);
  const double cy = std::cos(0.5 * yaw), sy = std::sin(0.5 * yaw);
  // q = qz(yaw) ⊗ qy(pitch) ⊗ qx(roll)
  Quat q;
  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;
  return qnormalize(q);
}

}  // namespace eskf_imu
