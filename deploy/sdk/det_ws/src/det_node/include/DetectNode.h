#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "YoloDet.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

class DetectNode : public rclcpp::Node
{
public:
    DetectNode();
    // 析构函数声明
    ~DetectNode() = default;

private:
    YoloDet yoloDet;
    // 成员变量
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    bool is_active_; // 控制发布状态的标志
    void timerCallback();
    void switchStateCallback(const std_msgs::msg::Bool::SharedPtr msg);
};

