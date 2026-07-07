#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <serial_driver/serial_driver.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <cstdio>   // sprintf 函数
#include <cstring>  // strlen 函数
#include <chrono>   // std::chrono 时间
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace rm_serial_driver
{
class RMSerialDriver : public rclcpp::Node
{
public:
  explicit RMSerialDriver(const rclcpp::NodeOptions & options);
  ~RMSerialDriver() override;

private:
  void getParams();
  void receiveData();
  // 核心：订阅你的火控指令话题
  void sendData(std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void reopenPort();

  // 串口硬件
  std::unique_ptr<IoContext> owned_ctx_;
  std::string device_name_;
  std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
  std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

  // 订阅火控指令
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr encoder_pub_;
  std::thread receive_thread_;
};
}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_