#pragma once
#include <Eigen/Dense>
class Simple3DKF
{
public:
  // 9维状态 [x,y,z, vx,vy,vz, ax,ay,az] 匀加速线性卡尔曼
  Eigen::Vector<double,9> X;
  Eigen::Matrix<double,9,9> F;
  Eigen::Matrix<double,3,9> H;
  Eigen::Matrix<double,9,9> P, Q;
  Eigen::Matrix<double,3,3> R;
  double dt;

  Simple3DKF(double dt_ = 0.01) : dt(dt_)
  {
    F.setIdentity();
    updateFMatrix();

    // 观测只取xyz
    H.setZero();
    H(0,0) = 1;
    H(1,1) = 1;
    H(2,2) = 1;

    P.setIdentity() * 1.0;
    Q.setZero();
    R.setIdentity() * 0.005;
  }

  // 实时刷新转移矩阵F
  void updateFMatrix()
  {
    double d = dt;
    double d2 = 0.5 * d * d;
    // x
    F(0,3) = d;
    F(0,6) = d2;
    F(3,6) = d;
    // y
    F(1,4) = d;
    F(1,7) = d2;
    F(4,7) = d;
    // z
    F(2,5) = d;
    F(2,8) = d2;
    F(5,8) = d;
  }

  // 三参数：位置噪声、速度噪声、加速度噪声
  void setNoise(double q_pos, double q_vel, double q_acc, double r_obs)
  {
    Q.setZero();
    Q(0,0)=Q(1,1)=Q(2,2) = q_pos;
    Q(3,3)=Q(4,4)=Q(5,5) = q_vel;
    Q(6,6)=Q(7,7)=Q(8,8) = q_acc;
    R.setIdentity() * r_obs;
  }

  // 初始化位置，速度加速度默认0
  void init(double tx0, double ty0, double tz0)
  {
    X.setZero();
    X(0)=tx0; X(1)=ty0; X(2)=tz0;
  }

  void predict()
  {
    updateFMatrix();
    X = F * X;
    P = F * P * F.transpose() + Q;
  }

  void update(double ztx, double zty, double ztz)
  {
    Eigen::Vector<double,3> Z;
    Z << ztx, zty, ztz;
    auto K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
    X = X + K * (Z - H * X);
    P = (Eigen::Matrix<double,9,9>::Identity() - K * H) * P;
  }

  // 延迟补偿预测T秒后xyz（匀加速递推）
  Eigen::Vector3d predictFuture(double T)
  {
    double x = X(0) + X(3)*T + 0.5*X(6)*T*T;
    double y = X(1) + X(4)*T + 0.5*X(7)*T*T;
    double z = X(2) + X(5)*T + 0.5*X(8)*T*T;
    return Eigen::Vector3d(x,y,z);
  }
};