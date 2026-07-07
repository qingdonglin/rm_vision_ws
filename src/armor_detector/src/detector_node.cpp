#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "armor_detector/detector.hpp"
 #include "ament_index_cpp/get_package_share_directory.hpp"
#include "armor_detector/simple_3d_kf.hpp"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <cmath>
#include "std_msgs/msg/float64_multi_array.hpp"
#include "rclcpp/parameter.hpp"
#include "rclcpp/parameter_value.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.h>
using namespace cv;
using namespace std;
using namespace std::placeholders;

class ArmorDetectorNode : public rclcpp::Node
{
public:
    ArmorDetectorNode() : Node("armor_detector")
    {
        //system("v4l2-ctl -c auto_exposure=1");//设置曝光
        //system("v4l2-ctl -c exposure_time_absolute=60");

        // 检测模式参数：默认启用模型检测，可通过launch参数切换
        this->declare_parameter<bool>("use_model_detect", true);
        use_model_detect_ = this->get_parameter("use_model_detect").as_bool();

        // 第二步：根据读取到的模式，设置对应曝光(移到这里才会生效)
    if (!use_model_detect_) {
        // 传统灯条模式：固定低曝光
        system("v4l2-ctl -c auto_exposure=1");
        system("v4l2-ctl -c exposure_time_absolute=60");
        RCLCPP_INFO(this->get_logger(), "传统检测模式：已设置手动低曝光");
    } /*{
    // 模型模式：中等固定手动曝光
    system("v4l2-ctl -c auto_exposure=1");
    system("v4l2-ctl -c exposure_time_absolute=120"); // 推荐100~150之间微调
    RCLCPP_INFO(this->get_logger(), "模型检测模式：已设置中等固定曝光");
    }*/

        if (use_model_detect_) {
            try {
                auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
                std::string model_path = pkg_path + "/model/opt-1208-001.onnx";
                std::string classify_path = pkg_path + "/model/mlp.onnx";
                std::string label_path = pkg_path + "/model/label.txt";

                float conf_th = this->declare_parameter("model_conf_threshold", 0.95f);
                int top_k = this->declare_parameter("model_top_k", 128);
                float nms_th = this->declare_parameter("model_nms_threshold", 0.3f);

                if (!detector_.initModel(model_path, classify_path, label_path, "AUTO", conf_th, top_k, nms_th)) {
                    RCLCPP_ERROR(this->get_logger(), "OpenVINO模型初始化失败，自动回退传统检测模式");
                    use_model_detect_ = false;
                } else {
                    RCLCPP_INFO(this->get_logger(), "OpenVINO模型检测模式已启用");
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "模型加载异常: %s", e.what());
                use_model_detect_ = false;
            }
        }


        subscriber_ = image_transport::create_subscription(
            this, "/image_raw",
            bind(&ArmorDetectorNode::imageCallback, this, _1),
            "raw"
        );// 订阅摄像头图像 /image_raw,绑定imageCallback回调

        publisher_ = image_transport::create_publisher(this, "/armor_result");// 发布处理后的图像 /armor_result
        kf_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/kf_state", 10);//卡尔曼状态发布器
        // 声明5个动态参数，默认值匹配Simple6DKF初始值
        this->declare_parameter<double>("kf_q_pos", 1.0);
        this->declare_parameter<double>("kf_q_vel", 12.0);
        this->declare_parameter<double>("kf_q_acc", 40.0);
        this->declare_parameter<double>("kf_r_obs", 0.06);
        this->declare_parameter<double>("kf_T_delay", 0.003);

        // 参数修改回调函数
        auto param_cb = [this](const std::vector<rclcpp::Parameter>& params) -> rcl_interfaces::msg::SetParametersResult
        {
            (void)params;
            double qp = this->get_parameter("kf_q_pos").as_double();
            double qv = this->get_parameter("kf_q_vel").as_double();
            double qa = this->get_parameter("kf_q_acc").as_double();
            double rr = this->get_parameter("kf_r_obs").as_double();
            
            kf.setNoise(qp, qv, qa, rr);

            rcl_interfaces::msg::SetParametersResult res;
            res.successful = true;
            res.reason = "update ok";
            return res;
        };
        this->add_on_set_parameters_callback(param_cb);
        //全局状态变量初始化
        kf_init = false;
        lost_count = 0;
        max_lost = 15;// 最多丢15帧还继续跟踪
        last_angle = 0.0;// 上一帧装甲角度
        last_w = 100.0;// 上一帧装甲宽度
        last_h = 30.0;// 上一帧装甲高度
        last_stamp = this->now();// 上一帧时间戳
        // 启动立刻同步Foxglove默认参数给卡尔曼
        double qp = this->get_parameter("kf_q_pos").as_double();
        double qv = this->get_parameter("kf_q_vel").as_double();
        double qa = this->get_parameter("kf_q_acc").as_double();
        double rr = this->get_parameter("kf_r_obs").as_double();
        kf.setNoise(qp, qv, qa, rr);
        //火控目标消息发布器
        target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>("/track_target", 10);
        // 初始化TF2监听组件，绑定节点时钟
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    // 接收图像消息，转换为OpenCV Mat格式
    cv::Mat img;
    try {
        img = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (...) {
        return;
    }

    // 计算帧间隔dt，用于卡尔曼时间步长
    rclcpp::Time curr_stamp = msg->header.stamp;
    double dt = (curr_stamp - last_stamp).seconds();
    last_stamp = curr_stamp;
    // 异常时间值钳位，防止超大/负dt破坏滤波
    if (dt > 0.1 || dt < 0) dt = 0.01;
    kf.dt = dt;

    // 图像识别流水线：颜色阈值→形态学去噪→提取灯条→匹配装甲
     std::vector<Light> lights;
     std::vector<Armor> armors;

     if (use_model_detect_) {
         armors = detector_.detectWithModel(img);
     } else {
         // === 你原有的传统检测代码，一个字没改 ===
         Mat mask = detector_.getColorMask(img);
         Mat clean = detector_.denoise(mask);
         imshow("1. Binary Mask (二值化结果)", mask);
         imshow("2. Clean Mask (去噪后结果)", clean);
         lights = detector_.findLights(img, clean);
         armors = detector_.matchLights(lights);
     }

    Point2f track_pt;       // 滤波后目标像素中心点（画红框用）
    bool draw = false;      // 标记是否需要绘制跟踪框
    // 提前全局初始化PoseResult，扩大作用域，保证drawResults始终有合法入参
    PoseResult res{};

    // 识别到有效装甲时执行PnP解算+三维卡尔曼滤波
    if (!armors.empty())
    {
        auto& armor = armors[0];
        // 单块装甲PnP解算，获取yaw/pitch/距离/相机三维xyz坐标
        res = detector_.solvePnPAndGetPose(armor.pts[0], armor.pts[1], armor.pts[2], armor.pts[3]);
        // PNP原始输出：OpenCV 右手光学系（X向右、Y向下、Z向前）
        double x_ocv = res.cam_xyz.x;
        double y_ocv = res.cam_xyz.y;
        double z_ocv = res.cam_xyz.z;

        // 右手系轴重映射，把pnp解算结果扳回与相机系平行的方向，把pnp结果，坐标和角度等搬到三维正常能理解的结果
        // camera_link 约定：X向前、Y向左、Z向上（和云台系轴向完全统一）
        double x_cam =  z_ocv;   // 光轴向前 = 相机连杆X正方向
        double y_cam = -x_ocv;   // 图像向右 = 连杆Y负方向（左为正）
        double z_cam = -y_ocv;   // 图像向下 = 连杆Z负方向（上为正）

        // ========== 再做TF转云台坐标系 ==========
        geometry_msgs::msg::PointStamped cam_pt, gim_pt;
        cam_pt.header.frame_id = "camera_link";
        cam_pt.header.stamp = msg->header.stamp;
        cam_pt.point.x = x_cam;
        cam_pt.point.y = y_cam;
        cam_pt.point.z = z_cam;
        double gx_obs, gy_obs, gz_obs;
        try {
            // 相机坐标系 → 云台gimbal_link坐标系
            gim_pt = tf_buffer_->transform(cam_pt, "gimbal_link", tf2::durationFromSec(0.01));
            gx_obs = gim_pt.point.x;
            gy_obs = gim_pt.point.y;
            gz_obs = gim_pt.point.z;
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "TF转换失败,临时使用相机坐标: %s", ex.what());
            // 兜底：用相机右手系坐标临时替代（保证编译通过，TF正常后不会走到这里）
            gx_obs = x_cam;
            gy_obs = y_cam;
            gz_obs = z_cam;
        }

        // 首次识别：用云台坐标初始化3D卡尔曼
        if (!kf_init)
        {
            kf.init(gx_obs, gy_obs, gz_obs);
            kf_init = true;
        }

        // 卡尔曼标准流程：先预测、再用云台观测值更新
        kf.predict();
        kf.update(gx_obs, gy_obs, gz_obs);

        // 延迟补偿：预测T毫秒后的云台系三维位置
        double T = this->get_parameter("kf_T_delay").as_double();
        Eigen::Vector3d gim_smooth = kf.predictFuture(T);

        // ========== 绘图用：云台系转回相机系，再反投影到像素 ==========
        geometry_msgs::msg::PointStamped gim_smooth_pt, cam_smooth_pt;
        gim_smooth_pt.header.frame_id = "gimbal_link";
        gim_smooth_pt.header.stamp = msg->header.stamp;
        gim_smooth_pt.point.x = gim_smooth(0);
        gim_smooth_pt.point.y = gim_smooth(1);
        gim_smooth_pt.point.z = gim_smooth(2);

        double cam_sx, cam_sy, cam_sz;
        try {
            cam_smooth_pt = tf_buffer_->transform(gim_smooth_pt, "camera_link", tf2::durationFromSec(0.01));
            double x_cam = cam_smooth_pt.point.x;
            double y_cam = cam_smooth_pt.point.y;
            double z_cam = cam_smooth_pt.point.z;

            // 逆转换：camera_link右手系 → OpenCV左手光学系
            double x_ocv_proj = -y_cam;
            double y_ocv_proj = -z_cam;
            double z_ocv_proj =  x_cam;

            cam_sx = x_ocv_proj;
            cam_sy = y_ocv_proj;
            cam_sz = z_ocv_proj;
        } catch (...) {
            // 转换失败兜底：直接用原始PNP结果投影
            cam_sx = x_ocv;
            cam_sy = y_ocv;
            cam_sz = z_ocv;
        }

        // 用OpenCV光学系坐标 + 内参 反投影到像素
        double fx = 650.0, fy = 650.0, cx = 320.0, cy = 240.0;
        double proj_u = fx * cam_sx / cam_sz + cx;
        double proj_v = fy * cam_sy / cam_sz + cy;

        
        track_pt = Point2f(proj_u, proj_v);

        // 发布卡尔曼全部状态话题（现在全部是云台坐标系数值）可以给foxglove画曲线和调参数用
        std_msgs::msg::Float64MultiArray msg_kf;
        msg_kf.data = {
            kf.X(0),kf.X(1),kf.X(2),
            kf.X(3),kf.X(4),kf.X(5),
            kf.X(6),kf.X(7),kf.X(8),
            gim_smooth(0),gim_smooth(1),gim_smooth(2)
        };
        kf_pub_->publish(msg_kf);

        lost_count = 0;
        draw = true;

        // 记录装甲像素尺寸、倾斜角度（和原来一致，不动)
        if (use_model_detect_) {
             last_w = cv::norm(armor.pts[0] - armor.pts[1]);
             last_h = cv::norm(armor.pts[0] - armor.pts[3]);
         } else {
             last_w = norm(armor.left_light.center - armor.right_light.center);
             last_h = (armor.left_light.length + armor.right_light.length) * 0.5f;
         }
        // 向量：从左灯指向右灯（对应装甲宽度方向，和RotatedRect的width边定义一致）
        Point2f d;
        if (use_model_detect_) d = armor.pts[1] - armor.pts[0];
        else d = armor.right_light.center - armor.left_light.center;
        // 图像坐标系Y轴向下，atan2结果天然是顺时针角度，直接转成度即可
        last_angle = atan2(d.y, d.x) * 180.0 / CV_PI;
        // 归一化到 0~180° 范围，完全匹配OpenCV RotatedRect的角度约定
        if (last_angle < 0) last_angle += 180.0f;

        // ========== 填充电控Target消息：直接输出云台坐标 ==========
        auto_aim_interfaces::msg::Target tgt_msg;
        tgt_msg.header.stamp = this->now();
        tgt_msg.header.frame_id = "gimbal_link"; // 坐标系改为云台
        tgt_msg.tracking = true;
        tgt_msg.armors_num = armors.size();
        // 卡尔曼平滑位置（天然云台坐标系，无需二次转换）
        tgt_msg.position.x = kf.X(0);
        tgt_msg.position.y = kf.X(1);
        tgt_msg.position.z = kf.X(2);
        // 速度（云台坐标系下物理速度，电控直接用）
        tgt_msg.velocity.x = kf.X(3);
        tgt_msg.velocity.y = kf.X(4);
        tgt_msg.velocity.z = kf.X(5);
        tgt_msg.yaw = res.yaw;//注意，这里的角度仅仅只是装甲板系的角度，火控只用到距离，用公式算出云台要旋转的角度
        tgt_msg.pitch = res.pitch;
        // 无陀螺角速度、半径、高度全部置0
        tgt_msg.v_yaw = 0.0;
        tgt_msg.radius_1 = 0.0;
        tgt_msg.radius_2 = 0.0;
        tgt_msg.dz = 0.0;
        target_pub_->publish(tgt_msg);
        
    }
    else
    {
        // 无新装甲识别，丢帧计数+1，仅执行卡尔曼预测外推
        if (kf_init && lost_count < max_lost)
        {
            kf.predict();
            lost_count++;
            draw = true;

            // ========== KF是云台系，转回相机系再投影到像素 ==========
            geometry_msgs::msg::PointStamped gim_pred_pt, cam_pred_pt;
            gim_pred_pt.header.frame_id = "gimbal_link";
            gim_pred_pt.header.stamp = msg->header.stamp;
            gim_pred_pt.point.x = kf.X(0);
            gim_pred_pt.point.y = kf.X(1);
            gim_pred_pt.point.z = kf.X(2);
            double cam_px, cam_py, cam_pz;
            try {
                cam_pred_pt = tf_buffer_->transform(gim_pred_pt, "camera_link", tf2::durationFromSec(0.01));
                double x_cam_pred = cam_pred_pt.point.x;
                double y_cam_pred = cam_pred_pt.point.y;
                double z_cam_pred = cam_pred_pt.point.z;

                // 逆转换：camera_link右手系 → OpenCV左手光学系
                cam_px = -y_cam_pred;
                cam_py = -z_cam_pred;
                cam_pz =  x_cam_pred;
            } catch (...) {
                // 兜底：默认3米正前方（光学系坐标）
                cam_px = 0.0;
                cam_py = 0.0;
                cam_pz = 3.0;
            }

            // 用光学系坐标 + 内参 反投影到像素
            double fx = 650.0, fy = 650.0, cx = 320.0, cy = 240.0;
            double proj_u = fx * cam_px / cam_pz + cx;
            double proj_v = fy * cam_py / cam_pz + cy;
            
            track_pt = Point2f(proj_u, proj_v);

            // 丢帧依旧发布卡尔曼外推跟踪数据（云台坐标系）
            auto_aim_interfaces::msg::Target tgt_msg;//自定义 ROS2 消息
            tgt_msg.header.stamp = this->now();
            tgt_msg.header.frame_id = "gimbal_link";
            tgt_msg.tracking = true;
            tgt_msg.armors_num = 0;
            tgt_msg.position.x = kf.X(0);
            tgt_msg.position.y = kf.X(1);
            tgt_msg.position.z = kf.X(2);
            tgt_msg.velocity.x = kf.X(3);
            tgt_msg.velocity.y = kf.X(4);
            tgt_msg.velocity.z = kf.X(5);
            tgt_msg.yaw = 0.0;
            tgt_msg.v_yaw = 0.0;
            tgt_msg.radius_1 = 0.0;
            tgt_msg.radius_2 = 0.0;
            tgt_msg.dz = 0.0;
            target_pub_->publish(tgt_msg);
        }

        // ========== 丢帧超过上限，通知火控目标丢失 ==========
        else if (kf_init && lost_count >= max_lost)
        {
            kf_init = false; // 重置跟踪状态
            auto_aim_interfaces::msg::Target tgt_msg;
            tgt_msg.header.stamp = this->now();
            tgt_msg.header.frame_id = "gimbal_link";
            tgt_msg.tracking = false; // 标记丢失
            target_pub_->publish(tgt_msg);
        }
        
    }


    // 绘制跟踪红框与滤波坐标文字
    if (draw)
    {
        RotatedRect rbox;
        // 当前帧识别到装甲：本帧实时尺寸+角度，和绿框姿态完全同步，红框运用卡尔曼的地方是装甲板中心点
        if (!armors.empty())
        {
            rbox = RotatedRect(track_pt, Size2f(last_w, last_h), last_angle);
        }
        else
        {
            // 丢帧跟踪，只能沿用历史姿态
            rbox = RotatedRect(track_pt, Size2f(last_w, last_h), last_angle);
        }

        Point2f pts[4];
        rbox.points(pts);
        for (int i = 0; i < 4; i++)
            line(img, pts[i], pts[(i+1)%4], Scalar(0,0,255), 2);
        circle(img, track_pt, 6, Scalar(0,0,255), -1);

        char text[128];
        if(kf_init)
        {
            // 打印滤波平滑后的云台三维米坐标
            sprintf(text, "Gimbal tx:%.2f ty:%.2f tz:%.2f", kf.X(0), kf.X(1), kf.X(2));
        }else{
            sprintf(text, "3DKF Not Init");
        }
        putText(img, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,255), 2);
            }

    // 绘制灯条、装甲框、角度文字
     if (use_model_detect_)
         detector_.drawModelResults(img, armors, res);
     else
         detector_.drawResults(img, lights, armors, res);
    // ====== 新增：弹出本地窗口显示最终检测结果 ======
    cv::imshow("检测结果", img);
    cv::waitKey(1);  // 必须加，否则窗口卡死不刷新
    // 把绘制完成的图像发布出去
    auto out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
    publisher_.publish(*out);
}
    //类私有成员变量
    image_transport::Subscriber subscriber_;
    image_transport::Publisher publisher_;
    bool use_model_detect_;  // true=模型检测，false=传统检测
    Detector detector_;
    Simple3DKF kf;
    bool kf_init;
    int lost_count, max_lost;
    float last_angle, last_w, last_h;
    rclcpp::Time last_stamp;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr kf_pub_;// 卡尔曼曲线发布
    rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;
    // TF2坐标变换缓存与监听器
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}