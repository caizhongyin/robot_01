#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class Talker : public rclcpp::Node
{
public:
    Talker() : Node("talker"), count_(0)
    {
        // 创建发布者，发布std_msgs::msg::String类型的消息到'topic'话题，队列大小为10
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        // 创建定时器，每500毫秒触发一次，调用timer_callback函数
        timer_ = this->create_wall_timer(
            500ms, std::bind(&Talker::timer_callback, this));
        RCLCPP_INFO(this->get_logger(), "发布者节点已启动!");
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello, world! " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "发布: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Talker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}