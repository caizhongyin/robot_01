#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using std::placeholders::_1;

class Listener : public rclcpp::Node
{
public:
    Listener() : Node("listener")
    {
        // 创建订阅者，订阅'topic'话题的std_msgs::msg::String类型消息，队列大小为10
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10, std::bind(&Listener::topic_callback, this, _1));
        RCLCPP_INFO(this->get_logger(), "订阅者节点已启动!");
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
        RCLCPP_INFO(this->get_logger(), "听到: '%s'", msg->data.c_str());
    }
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Listener>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}