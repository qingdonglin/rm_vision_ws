#include "rclcpp/rclcpp.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>
#include <chrono>

class FireControlNode : public rclcpp::Node
{
public:
    FireControlNode() : Node("fire_control")
    {
        // 可动态调参，默认值适配RM步兵常规配置
        this->declare_parameter<double>("bullet_speed", 28.0);    // 弹速 m/s
        this->declare_parameter<double>("gravity", 9.8);         // 重力加速度 m/s²
        this->declare_parameter<double>("fire_yaw_thresh", 2.0); // 偏航对准阈值（度）
        this->declare_parameter<double>("fire_pitch_thresh", 2.0);// 俯仰对准阈值（度）
        this->declare_parameter<int>("ballistic_iter", 3);       // 弹道迭代次数

        // 订阅视觉跟踪结果
        target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
            "/track_target", 10,
            std::bind(&FireControlNode::targetCallback, this, std::placeholders::_1)
        );

        // 发布云台绝对角度控制指令：[目标yaw(rad), 目标pitch(rad), 开火信号(0/1)]
        gimbal_cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/gimbal_cmd", 10);

        // TF监听，获取云台实时真实绝对姿态（保留备用，仿真主要用编码器）
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        //订阅云台编码器角度（仿真模拟话题） 
        encoder_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/gimbal_encoder", 10,
            std::bind(&FireControlNode::encoderCallback, this, std::placeholders::_1)
        );
        

        // 10ms定时发布指令，保证信号持续
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&FireControlNode::publishCurrentCmd, this)
        );

        // 初始化缓存默认值
        last_yaw_ = 0.0;
        last_pitch_ = 0.0;
        last_fire_ = 0;

        RCLCPP_INFO(this->get_logger(), "绝对角度火控节点启动完成");
    }

private:
    // 定时发布缓存的最新指令
    void publishCurrentCmd()
    {
        publishCommand(last_yaw_, last_pitch_, last_fire_);
    }

    // ========== 新增：编码器角度回调函数 ==========
    // 符号转换规则和 gimbal_tf_broadcaster 保持完全一致
    void encoderCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 2) return;
        // 仿真约定：右转yaw正、抬头pitch正 → 转右手系（左转正、低头正）根据电控定
        curr_yaw_   = msg->data[0];
        curr_pitch_ = msg->data[1];
    }
    // =============================================

    void targetCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg)
    {
        // 丢失目标：禁射，角度指令保持上一帧（避免云台突然回零）
        if (!msg->tracking)
        {
            last_fire_ = 0;
            return;
        }

        // 读取参数
        double v_bullet = this->get_parameter("bullet_speed").as_double();
        double g = this->get_parameter("gravity").as_double();
        int iter_times = this->get_parameter("ballistic_iter").as_int();

        // 云台坐标系下目标状态
        double x = msg->position.x;
        double y = msg->position.y;
        double z = msg->position.z;
        double vx = msg->velocity.x;
        double vy = msg->velocity.y;
        double vz = msg->velocity.z;

        // ========== 1. 弹道迭代解算 → 得到【相对云台的偏角】 ==========
        double t_flight = std::sqrt(x*x + y*y + z*z) / v_bullet;
        double tar_yaw_rel, tar_pitch_rel; // 后缀 _rel = 云台系相对角度

        for (int i = 0; i < iter_times; i++)
        {
            double x_pred = x + vx * t_flight;
            double y_pred = y + vy * t_flight;
            double z_pred = z + vz * t_flight;

            double d_horiz = std::sqrt(x_pred * x_pred + y_pred * y_pred);
            d_horiz = std::max(d_horiz, 0.01);

            double drop = 0.5 * g * t_flight * t_flight;
            double z_corr = z_pred + drop;

            tar_yaw_rel = std::atan2(y_pred, x_pred);
            // 右手系TF中pitch正方向为低头，几何解算抬头为正，加负号对齐基准
            tar_pitch_rel = -std::atan2(z_corr, d_horiz);

            t_flight = d_horiz / (v_bullet * std::cos(tar_pitch_rel));
        }

        // ========== 2. 读取当前云台绝对角度（直接读编码器缓存，无TF延迟） ==========
        double real_yaw = curr_yaw_;
        double real_pitch = curr_pitch_;
        bool angle_valid = true; // 仿真环境默认角度有效

        // ========== 3. 合成目标绝对角度 + 开火判定 + 更新指令 ==========
        if (angle_valid)
        {
            // 目标绝对角度 = 云台真实绝对角度 + 目标相对云台偏角
            double tar_yaw_abs = normalizeAngle(real_yaw + tar_yaw_rel);
            double tar_pitch_abs = normalizeAngle(real_pitch + tar_pitch_rel);

            // 绝对角度误差计算（归一化处理±π边界）
            double err_yaw = std::fabs(normalizeAngle(tar_yaw_abs - real_yaw));
            double err_pitch = std::fabs(normalizeAngle(tar_pitch_abs - real_pitch));

            // 阈值转换,转弧度
            double th_yaw = this->get_parameter("fire_yaw_thresh").as_double() * M_PI / 180.0;
            double th_pitch = this->get_parameter("fire_pitch_thresh").as_double() * M_PI / 180.0;
            // 新增：打印误差和阈值，转成度方便直观查看
            RCLCPP_INFO(this->get_logger(), "对准误差 yaw:%.2f° pitch:%.2f° | 开火阈值 %.1f°",
                err_yaw * 180 / M_PI, err_pitch * 180 / M_PI,
                this->get_parameter("fire_yaw_thresh").as_double());
            // 开火判定
            last_fire_ = (err_yaw < th_yaw && err_pitch < th_pitch) ? 1 : 0;

            // 缓存绝对角度指令，下发给电控
            last_yaw_ = tar_yaw_abs;
            last_pitch_ = tar_pitch_abs;
        }
        else
        {
            // TF获取失败：强制禁射，角度指令保持上一帧有效值
            last_fire_ = 0;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "云台TF失效,已禁射并保持上一帧姿态");
        }
    }

    // 读取云台真实姿态（base_link 系绝对角度）
    bool getGimbalRealAngle(double & yaw, double & pitch)
    {
        try {
            auto tf = tf_buffer_->lookupTransform(
                "base_link", "gimbal_link",
                tf2::TimePointZero,
                tf2::durationFromSec(0.01)
            );
            tf2::Quaternion q(
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w
            );
            double roll;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
            return true;
        } catch (const tf2::TransformException & ex) {
            return false;
        }
    }

    // 角度归一化到 [-π, π]
    double normalizeAngle(double angle)
    {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    }

    // 封装发布控制指令
    void publishCommand(double yaw, double pitch, int fire)
    {
        std_msgs::msg::Float64MultiArray cmd_msg;
        // 右手系绝对角度 → 电控原始约定（右转yaw正、抬头pitch正）
        double cmd_yaw = yaw;
        double cmd_pitch = pitch;
        cmd_msg.data = {cmd_yaw, cmd_pitch, static_cast<double>(fire)};
        gimbal_cmd_pub_->publish(cmd_msg);
    }

    // 订阅器、发布器、TF组件
    rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr target_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gimbal_cmd_pub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ========== 新增：编码器话题订阅 + 当前云台角度缓存 ==========
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr encoder_sub_;
    double curr_yaw_ = 0.0;   // 当前云台yaw（右手系，弧度）
    double curr_pitch_ = 0.0; // 当前云台pitch（右手系，弧度）

    // 缓存最新指令 + 定时器（全部为底盘系绝对角度）
    double last_yaw_;
    double last_pitch_;
    int last_fire_;
    rclcpp::TimerBase::SharedPtr timer_;

};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FireControlNode>());
    rclcpp::shutdown();
    return 0;
}