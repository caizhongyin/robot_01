# det_ws
## 创建工作空间目录
```bash
mkdir -p ~/det_ws/src
cd ~/det_ws
ros2 pkg create --build-type ament_cmake det_node --dependencies rclcpp std_msgs
```
## 编译
```bash
colcon build --packages-select det_node
```
## 启动
```bash
source ./install/setup.bash
ros2 run det_node DetectNode
```
## 状态开关
```bash
ros2 topic pub /switch_state std_msgs/msg/Bool "{data: true}" -1
ros2 topic pub /switch_state std_msgs/msg/Bool "{data: false}" -1
```
## test ros2
```bash
cmd1:
source ~/det_ws/install/setup.bash
ros2 run det_node talker

cmd2:
source ~/det_ws/install/setup.bash
ros2 run det_node listener
```
