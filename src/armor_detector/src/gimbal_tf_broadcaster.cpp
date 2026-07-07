#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
// 编码器消息头文件
#include "std_msgs/msg/float64_multi_array.hpp"
#include <cmath>
#include <chrono>
class GimbalTfBroadcaster : public rclcpp::Node
{
public:
    GimbalTfBroadcaster() : Node("gimbal_tf_broadcaster")//订阅/gimbal_encoder获取云台 yaw/pitch 角度；固定 50Hz 定时组装 base_link→gimbal_link 坐标变换；广播变换给全系统所有需要坐标转换的节点（视觉、RViz、调试工具）。
    {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);//TF 专用发布句柄，不用手动创建 Publisher，自动往/tf发数据
        //底盘base_link原点 → 云台旋转轴心gimbal_link的固定机械偏移，只和装配尺寸有关
        base_to_gimbal_x_ = 0.0;
        base_to_gimbal_y_ = 0.0;
        base_to_gimbal_z_ = 0.3;
        gimbal_yaw_ = 0.0;
        gimbal_pitch_ = 0.0;

        // 编码器订阅
        try{
            rclcpp::QoS encoder_qos(10);
            encoder_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
                "/gimbal_encoder",
                encoder_qos,
                std::bind(&GimbalTfBroadcaster::encoderCallback, this, std::placeholders::_1)
            );
            RCLCPP_INFO(this->get_logger(), "✅ 已创建编码器话题订阅: /gimbal_encoder");
        }catch(std::exception &e){
            RCLCPP_ERROR(this->get_logger(), "❌ 创建订阅失败: %s", e.what());
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&GimbalTfBroadcaster::publishTf, this)
        );

        RCLCPP_INFO(this->get_logger(), "云台动态TF广播节点已启动(实机订阅模式)");
    }

private:
    // 编码器角度回调
    void encoderCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        

        // ========== 改造2：直观确认消息有没有进回调 ==========
        RCLCPP_INFO(this->get_logger(), "收到编码器消息 | 数组长度:%zu | 数值:[%.3f, %.3f]",
                    msg->data.size(), msg->data[0], msg->data[1]);

        // 安全校验：数组长度不足直接丢弃，防止越界崩溃，必须至少 2 个元素（yaw、pitch）
        if (msg->data.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "编码器数据长度不足: %zu", msg->data.size());
            return;
        }
        // 电控发弧度直接赋值，若角度的话用下面
        // 右手系适配：电控约定（右转yaw正、抬头pitch正）→ 右手系旋转正方向，统一取反
        gimbal_yaw_ = msg->data[0];
        gimbal_pitch_ = msg->data[1];

        // 角度(°) 转 弧度(rad)
        // gimbal_yaw_ = msg->data[0] * M_PI / 180.0;
        // gimbal_pitch_ = msg->data[1] * M_PI / 180.0;

        // 方向不匹配时，对应加负号修正
        // gimbal_yaw_ = -msg->data[0];
        // gimbal_pitch_ = -msg->data[1];
    }

    void publishTf()
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->now();
        t.header.frame_id = "base_link";// 父坐标系：车身底盘
        t.child_frame_id = "gimbal_link";// 子坐标系：云台中心

        // 平移：固定安装尺寸
        t.transform.translation.x = base_to_gimbal_x_;
        t.transform.translation.y = base_to_gimbal_y_;
        t.transform.translation.z = base_to_gimbal_z_;

        // 旋转：云台角度转四元数（roll=0，pitch、yaw为实时角度
        tf2::Quaternion q;
        q.setRPY(0.0, gimbal_pitch_, gimbal_yaw_); 
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(t);//广播变换到全局 TF 树
    }

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr encoder_sub_;

    // 固定安装偏移量
    double base_to_gimbal_x_;
    double base_to_gimbal_y_;
    double base_to_gimbal_z_;

    // 云台实时角度（单位：弧度）
    double gimbal_yaw_;
    double gimbal_pitch_;

    
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GimbalTfBroadcaster>());
    rclcpp::shutdown();
    return 0;
}