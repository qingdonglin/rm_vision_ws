from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1、云台动态TF 最先启动
    gimbal_tf_node = Node(
        package="armor_detector",
        executable="gimbal_tf_broadcaster",
        name="gimbal_tf_broadcaster",
        output="screen"
    )

    # 相机静态TF：仅保留物理安装偏移，轴向已在代码里统一
    static_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            # x y z：相机相对云台旋转中心的物理安装位置（单位：米）
            "0.0",   # X轴：向前cm
            "0.00",  # Y轴：向右cm（右手系左为正，右为负）
            "0.02",    # Z轴：高度偏移
            # 欧拉角顺序：yaw pitch roll（ZYX外旋），仿真安装无偏转，全为0
            "0.0", "0.0", "0.0",
            "gimbal_link",  # 父坐标系：云台中心
            "camera_link"   # 子坐标系：相机光心
        ],
        output="screen"
    )

        # 3、串口通讯节点
    serial_config = os.path.join(
        get_package_share_directory('rm_serial_driver'), 'config', 'serial_driver.yaml')
    serial_node = Node(
        package='rm_serial_driver',
        executable='rm_serial_driver_node',
        output='screen',
        parameters=[serial_config],
    )

    # 3、视觉识别 延迟2秒启动，等TF就绪
    vision_detector = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="armor_detector",
                executable="armor_detector_node",
                output="screen",
                parameters=[
                    # ========== 核心切换开关 ==========
                    # true = AI模型检测，false = 传统灯条检测
                    {"use_model_detect":    False},

                    # ========== 模型检测调参 ==========
                    {"model_conf_threshold": 0.25},  # 置信度阈值，越大越严格
                    {"model_nms_threshold": 0.3},    # 去重阈值，越小去重越强
                    {"model_top_k": 128},            # 最大候选目标数

                    # ========== 卡尔曼原有参数 ==========
                    {"kf_q_pos": 1.0},
                    {"kf_q_vel": 12.0},
                    {"kf_q_acc": 40.0},
                    {"kf_r_obs": 0.06},
                    {"kf_T_delay": 0.003}
                ]
            )
        ]
    )

    # 4、火控解算 延迟3秒最后启动
    fire_ctrl_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="armor_detector",
                executable="fire_control_node",
                name="fire_control",
                output="screen"
            )
        ]
    )

    return LaunchDescription([
        gimbal_tf_node,
        static_camera_tf,
        serial_node,
        vision_detector,
        fire_ctrl_node
    ])