#include <memory>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <linkattacher_msgs/srv/attach_link.hpp>
#include <linkattacher_msgs/srv/detach_link.hpp>

using namespace std::chrono_literals;

class VisionPickPlace : public rclcpp::Node
{

public:

VisionPickPlace()
:Node("vision_pick_place_node"),
 move_group(std::shared_ptr<rclcpp::Node>(this),"ur5_manipulator"),
 gripper(std::shared_ptr<rclcpp::Node>(this),"robotiq_gripper")
{

move_group.setPoseReferenceFrame("base_link");

tf_buffer_=std::make_shared<tf2_ros::Buffer>(this->get_clock());

tf_listener_=std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

attach_client_=this->create_client<linkattacher_msgs::srv::AttachLink>("/ATTACHLINK");

detach_client_=this->create_client<linkattacher_msgs::srv::DetachLink>("/DETACHLINK");

rclcpp::sleep_for(3s);

execute();
}

private:

void openGripper()
{

gripper.setJointValueTarget("robotiq_85_left_knuckle_joint",0.0);

gripper.move();
}

void closeGripper()
{

gripper.setJointValueTarget("robotiq_85_left_knuckle_joint",0.25);

gripper.move();
}

geometry_msgs::msg::Pose getObjectPose()
{

auto tf=tf_buffer_->lookupTransform(
"base_link",
"object_grasp_frame",
tf2::TimePointZero);

geometry_msgs::msg::Pose pose;

pose.position.x=tf.transform.translation.x;
pose.position.y=tf.transform.translation.y;
pose.position.z=tf.transform.translation.z;

pose.orientation=tf.transform.rotation;

return pose;
}

void moveToPose(geometry_msgs::msg::Pose pose)
{

move_group.setPoseTarget(pose);

moveit::planning_interface::MoveGroupInterface::Plan plan;

bool success=
(move_group.plan(plan)==moveit::core::MoveItErrorCode::SUCCESS);

if(success)
move_group.move();
}

void attachObject()
{

auto req=
std::make_shared<linkattacher_msgs::srv::AttachLink::Request>();

req->model1_name="cobot";
req->link1_name="wrist_3_link";

req->model2_name="green_cube";
req->link2_name="link_1";

attach_client_->wait_for_service();

attach_client_->async_send_request(req);
}

void detachObject()
{

auto req=
std::make_shared<linkattacher_msgs::srv::DetachLink::Request>();

req->model1_name="cobot";
req->link1_name="wrist_3_link";

req->model2_name="green_cube";
req->link2_name="link_1";

detach_client_->wait_for_service();

detach_client_->async_send_request(req);
}

void execute()
{

openGripper();

auto grasp=getObjectPose();

geometry_msgs::msg::Pose pregrasp=grasp;

pregrasp.position.z+=0.12;

moveToPose(pregrasp);

moveToPose(grasp);

closeGripper();

attachObject();

moveToPose(pregrasp);

geometry_msgs::msg::Pose place;

place.position.x=0.35;
place.position.y=0.55;
place.position.z=0.45;

place.orientation.w=1.0;

moveToPose(place);

openGripper();

detachObject();

RCLCPP_INFO(this->get_logger(),"Pick and Place Completed");
}

moveit::planning_interface::MoveGroupInterface move_group;
moveit::planning_interface::MoveGroupInterface gripper;

std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

rclcpp::Client<linkattacher_msgs::srv::AttachLink>::SharedPtr attach_client_;
rclcpp::Client<linkattacher_msgs::srv::DetachLink>::SharedPtr detach_client_;

};

int main(int argc,char** argv)
{

rclcpp::init(argc,argv);

auto node=std::make_shared<VisionPickPlace>();

rclcpp::spin(node);

rclcpp::shutdown();

return 0;
}