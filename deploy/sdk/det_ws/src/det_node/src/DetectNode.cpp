#include "DetectNode.h"

DetectNode::DetectNode()
: Node("_detect"),
  is_active_(false)
{
  yoloDet.init();
  // 创建发布者
  publisher_ = this->create_publisher<std_msgs::msg::String>("output_message", 10);
  
  // 创建订阅者
  subscription_ = this->create_subscription<std_msgs::msg::Bool>("switch_state", 10, std::bind(&DetectNode::switchStateCallback, this, _1));
  
  // 创建定时器，但初始时不激活, 使用100ms的间隔
  timer_ = this->create_wall_timer(100ms, std::bind(&DetectNode::timerCallback, this));
  
  // 初始时停止定时器
  timer_->cancel();
  
  RCLCPP_INFO(this->get_logger(), "DetectNode 节点已启动");
  RCLCPP_INFO(this->get_logger(), "初始状态: 不发布内容");
  RCLCPP_INFO(this->get_logger(), "等待 /switch_state 话题的开关指令...");
}

void DetectNode::switchStateCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (msg->data) {
    // 收到开启指令
    if (!is_active_) {
      is_active_ = true;
      timer_->reset(); // 启动/重启定时器
      RCLCPP_INFO(this->get_logger(), "收到开启指令，开始以0.2秒间隔发布内容");
    } else {
      RCLCPP_INFO(this->get_logger(), "重复收到开启指令，保持发布状态");
    }
  } else {
    // 收到关闭指令
    if (is_active_) {
      is_active_ = false;
      timer_->cancel(); // 停止定时器
      RCLCPP_INFO(this->get_logger(), "收到关闭指令，停止发布内容");
    } else {
      RCLCPP_INFO(this->get_logger(), "重复收到关闭指令，保持停止状态");
    }
  }
}

void DetectNode::timerCallback()
{
  if (is_active_) {
    cv::Mat img = cv::imread(R"(/home/caizhongyin/code/sdk/trt/test_trt/images/test_det.jpg)");
    //std::cout << img.size() << std::endl;
    auto t1 = cv::getTickCount();
    yoloDet.inference(img);
    auto t2 = cv::getTickCount();
    std::string cost_time = cv::format("yolo time: %.2f ms", (t2 - t1) / cv::getTickFrequency() * 1000);
    std::cout << cost_time << std::endl;
    // 创建并发布消息
    auto message = std_msgs::msg::String();
    message.data = "正在发布，时间: " + std::to_string(this->now().seconds());
    publisher_->publish(message);
    
    // 可选：添加一些日志输出，但注意不要太频繁以免影响性能
    static int count = 0;
    if (++count % 5 == 0) { // 每5次发布输出一次日志
      RCLCPP_INFO(this->get_logger(), "已发布: %s", message.data.c_str());
    }
  }
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectNode>());
  rclcpp::shutdown();
  return 0;
}