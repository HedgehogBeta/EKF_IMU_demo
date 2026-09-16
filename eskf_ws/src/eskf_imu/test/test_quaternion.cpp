// M2 单元测试：手写 quaternion.hpp 与 Eigen 对照 + 手算用例。
// 不依赖 ROS，纯 C++ 可执行。plan.md 陷阱 3：分量顺序错会表现为误差恒 180°，
// 这里先用"单位四元数 ⊗ 90° 绕 z"的手算值把分量顺序钉死。
#include <cmath>
#include <cstdio>
#include <random>

#include <Eigen/Geometry>

#include "eskf_imu/quaternion.hpp"

using eskf_imu::Mat3;
using eskf_imu::Quat;
using eskf_imu::Vec3;

static int failures = 0;

#define CHECK_NEAR(expr, expected, tol, msg)                             \
  do {                                                                   \
    double e_ = (expr), x_ = (expected);                                 \
    if (!(std::abs(e_ - x_) <= (tol))) {                                 \
      std::printf("FAIL %s:%d %s: got %.12f want %.12f (tol %.1e)\n",    \
                  __FILE__, __LINE__, (msg), e_, x_, static_cast<double>(tol)); \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

static Eigen::Quaterniond to_eigen(const Quat& q) {
  // Eigen 构造是 (w,x,y,z)，内部存储 [x,y,z,w] —— 手写结构体 w 在前，别混
  return Eigen::Quaterniond(q.w, q.x, q.y, q.z);
}

// 随机单位四元数
static Quat random_quat(std::mt19937& gen) {
  std::normal_distribution<double> d(0.0, 1.0);
  Quat q{d(gen), d(gen), d(gen), d(gen)};
  return eskf_imu::qnormalize(q);
}

static void test_known_values() {
  // 手算：i ⊗ j = k，即 (0,1,0,0) ⊗ (0,0,1,0) = (0,0,0,1)
  Quat k = eskf_imu::qmul({0, 1, 0, 0}, {0, 0, 1, 0});
  CHECK_NEAR(k.w, 0, 1e-15, "i*j w");
  CHECK_NEAR(k.x, 0, 1e-15, "i*j x");
  CHECK_NEAR(k.y, 0, 1e-15, "i*j y");
  CHECK_NEAR(k.z, 1, 1e-15, "i*j z");

  // 恒等式：q ⊗ q* = 1
  std::mt19937 gen7(7);
  Quat q = random_quat(gen7);
  Quat id = eskf_imu::qmul(q, eskf_imu::qconj(q));
  CHECK_NEAR(id.w, 1, 1e-12, "q*q* w");
  CHECK_NEAR(id.x, 0, 1e-12, "q*q* x");

  // Exp(0) = 恒等
  Quat e0 = eskf_imu::qexp({0, 0, 0});
  CHECK_NEAR(e0.w, 1, 1e-15, "Exp(0) w");

  // 手算：Exp(90° 绕 z) = (cos45°, 0, 0, sin45°)
  Quat e90 = eskf_imu::qexp({0, 0, M_PI / 2});
  CHECK_NEAR(e90.w, std::sqrt(0.5), 1e-12, "Exp(90z) w");
  CHECK_NEAR(e90.z, std::sqrt(0.5), 1e-12, "Exp(90z) z");

  // 手算：Exp(180° 绕 x) = (0, 1, 0, 0)
  Quat e180 = eskf_imu::qexp({M_PI, 0, 0});
  CHECK_NEAR(e180.w, 0, 1e-12, "Exp(180x) w");
  CHECK_NEAR(e180.x, 1, 1e-12, "Exp(180x) x");
}

static void test_vs_eigen_random() {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> ang(-M_PI, M_PI);
  for (int i = 0; i < 200; ++i) {
    const Quat a = random_quat(gen), b = random_quat(gen);

    // 乘法
    Quat ab = eskf_imu::qmul(a, b);
    Eigen::Quaterniond ab_e = to_eigen(a) * to_eigen(b);
    CHECK_NEAR((ab_e.coeffs() - Eigen::Vector4d(ab.x, ab.y, ab.z, ab.w)).norm(), 0, 1e-12,
               "qmul vs Eigen");

    // Exp vs AngleAxis
    Vec3 phi{ang(gen), ang(gen), ang(gen)};
    Quat ex = eskf_imu::qexp(phi);
    const double norm_phi = phi.x * phi.x + phi.y * phi.y + phi.z * phi.z;
    Eigen::Quaterniond ex_e(
        Eigen::AngleAxisd(std::sqrt(norm_phi),
                          Eigen::Vector3d(phi.x, phi.y, phi.z).normalized()));
    // 符号对齐后比较（q 与 -q 等价）
    double s = (ex_e.w() * ex.w + ex_e.x() * ex.x + ex_e.y() * ex.y + ex_e.z() * ex.z) < 0 ? -1.0
                                                                                           : 1.0;
    CHECK_NEAR(std::abs(ex_e.w() - s * ex.w) + std::abs(ex_e.x() - s * ex.x) +
                   std::abs(ex_e.y() - s * ex.y) + std::abs(ex_e.z() - s * ex.z),
               0, 1e-12, "qexp vs Eigen");

    // toRotationMatrix vs Eigen，且作用于向量一致
    Mat3 R = eskf_imu::quat_to_rotation_matrix(a);
    Eigen::Matrix3d Re = to_eigen(a).toRotationMatrix();
    double dR = 0;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) dR += std::abs(R(r, c) - Re(r, c));
    CHECK_NEAR(dR, 0, 1e-12, "toRotationMatrix vs Eigen");

    Eigen::Vector3d v = Eigen::Vector3d::Random();
    Eigen::Vector3d v_mine = Eigen::Vector3d(
        R(0, 0) * v.x() + R(0, 1) * v.y() + R(0, 2) * v.z(),
        R(1, 0) * v.x() + R(1, 1) * v.y() + R(1, 2) * v.z(),
        R(2, 0) * v.x() + R(2, 1) * v.y() + R(2, 2) * v.z());
    CHECK_NEAR((Re * v - v_mine).norm(), 0, 1e-12, "R*v vs Eigen");

    // fromRotationMatrix 往返（Eigen 同样用 Shepperd，可直接对照）
    Quat back = eskf_imu::quat_from_rotation_matrix(R);
    Eigen::Quaterniond back_e = to_eigen(back);
    double sgn = back_e.dot(to_eigen(a)) < 0 ? -1.0 : 1.0;
    CHECK_NEAR((back_e.coeffs() - sgn * Eigen::Vector4d(a.x, a.y, a.z, a.w)).norm(), 0, 1e-12,
               "fromRotationMatrix vs Eigen");

    // 欧拉角往返（避开 pitch = ±90° 奇异区）
    const double roll = 0.8 * ang(gen) * 0.5, pitch = 0.9 * ang(gen) * 0.25, yaw = 0.8 * ang(gen) * 0.5;
    Quat qe = eskf_imu::quat_from_euler_zyx(roll, pitch, yaw);
    eskf_imu::EZYX e2 = eskf_imu::quat_to_euler_zyx(qe);
    CHECK_NEAR(e2.roll, roll, 1e-12, "euler roundtrip roll");
    CHECK_NEAR(e2.pitch, pitch, 1e-12, "euler roundtrip pitch");
    CHECK_NEAR(e2.yaw, yaw, 1e-12, "euler roundtrip yaw");

    // 静止比力 -> 欧拉 -> 旋转后应把 (0,0,1) 转回世界系竖直（yaw=0 时）
    Eigen::Vector3d f_e = Eigen::Vector3d(0.01, -0.02, 0.9997).normalized();
    eskf_imu::EZYX ea =
        eskf_imu::euler_from_specific_force({f_e.x(), f_e.y(), f_e.z()});
    Mat3 Ra = quat_to_rotation_matrix(eskf_imu::quat_from_euler_zyx(ea.roll, ea.pitch, 0.0));
    Eigen::Vector3d g_b(-f_e.x(), -f_e.y(), -f_e.z());  // f 的反向即重力方向
    Eigen::Vector3d g_w(Ra(0, 0) * g_b.x() + Ra(0, 1) * g_b.y() + Ra(0, 2) * g_b.z(),
                        Ra(1, 0) * g_b.x() + Ra(1, 1) * g_b.y() + Ra(1, 2) * g_b.z(),
                        Ra(2, 0) * g_b.x() + Ra(2, 1) * g_b.y() + Ra(2, 2) * g_b.z());
    CHECK_NEAR(g_w.x(), 0, 1e-9, "gravity aligned x");
    CHECK_NEAR(g_w.y(), 0, 1e-9, "gravity aligned y");
    CHECK_NEAR(g_w.z(), -1, 1e-9, "gravity aligned z");
  }
}

static void test_slerp() {
  std::mt19937 gen(11);
  // 端点：u=0 取 a，u=1 取 b（q 与 -q 等价，比较时统一对齐符号）
  const Quat a = random_quat(gen), b = random_quat(gen);
  auto same_rotation = [](const Quat& p, const Quat& q) {
    const double s = eskf_imu::qdot(p, q) < 0 ? -1.0 : 1.0;
    return std::abs(p.w - s * q.w) + std::abs(p.x - s * q.x) + std::abs(p.y - s * q.y) +
           std::abs(p.z - s * q.z);
  };
  CHECK_NEAR(same_rotation(eskf_imu::qslerp(a, b, 0.0), a), 0, 1e-12, "slerp u=0");
  CHECK_NEAR(same_rotation(eskf_imu::qslerp(a, b, 1.0), b), 0, 1e-12, "slerp u=1");

  // 随机对照：Eigen 的 slerp 同样走短弧（内部用 dot 的绝对值），所以随机对里
  // 大约一半的 dot 是负的，这一比就把符号展开也覆盖了。
  for (int i = 0; i < 200; ++i) {
    const Quat p = random_quat(gen), q = random_quat(gen);
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    const Quat mine = eskf_imu::qslerp(p, q, u);
    Eigen::Quaterniond e = to_eigen(p).slerp(u, to_eigen(q));
    const double s = (e.w() * mine.w + e.x() * mine.x + e.y() * mine.y + e.z() * mine.z) < 0
                         ? -1.0
                         : 1.0;
    CHECK_NEAR(std::abs(e.w() - s * mine.w) + std::abs(e.x() - s * mine.x) +
                   std::abs(e.y() - s * mine.y) + std::abs(e.z() - s * mine.z),
               0, 1e-12, "qslerp vs Eigen");
  }

  // 输入整体取反不应改变结果：q 与 -q 是同一旋转，取反后必须仍走短弧
  const Quat c = random_quat(gen), d = random_quat(gen);
  const Quat flipped{-d.w, -d.x, -d.y, -d.z};
  CHECK_NEAR(same_rotation(eskf_imu::qslerp(c, d, 0.37), eskf_imu::qslerp(c, flipped, 0.37)), 0,
             1e-12, "slerp 对 q -> -q 不敏感");

  // 实测数据里唯一那次符号翻转（pose_cov.csv 第 17957 -> 17958 帧，真转角 0.4705°）：
  // 两点几乎反向（点积 −0.9999916），插值中点必须在两点之间（≈0.2353°），
  // 而不是绕着长弧跑到 179.76° 外去。
  const Quat qa{-0.498322, -0.001200, 0.011233, 0.866919};
  const Quat qb{0.501876, 0.001140, -0.011313, -0.864865};
  const Quat mid = eskf_imu::qslerp(qa, qb, 0.5);
  const double dot_mid = eskf_imu::qdot(eskf_imu::qnormalize(qa), mid);
  const double ang_mid_deg = 2.0 * std::acos(std::abs(dot_mid)) * 180.0 / M_PI;
  CHECK_NEAR(ang_mid_deg, 0.2353, 5e-3, "翻转段插值中点应贴近端点（度）");
}

int main() {
  test_known_values();
  test_vs_eigen_random();
  test_slerp();
  if (failures == 0) {
    std::printf("test_quaternion: all checks passed\n");
    return 0;
  }
  std::printf("test_quaternion: %d failures\n", failures);
  return 1;
}
