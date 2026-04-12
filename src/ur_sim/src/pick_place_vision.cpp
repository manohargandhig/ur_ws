#include <memory>
#include <chrono>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include "ur_sim/msg/detected_object_array.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <linkattacher_msgs/srv/attach_link.hpp>
#include <linkattacher_msgs/srv/detach_link.hpp>

using namespace std::chrono_literals;

class MinimalPickPlace : public rclcpp::Node
{
public:
    MinimalPickPlace()
        : Node("minimal_pick_place"),
          move_group(std::shared_ptr<rclcpp::Node>(this), "ur5_manipulator")
    {
        move_group.setPoseReferenceFrame("base_link");

        // 🔥 ULTRA SAFE
        move_group.setMaxVelocityScalingFactor(0.02);
        move_group.setMaxAccelerationScalingFactor(0.02);
        move_group.setPlanningTime(10.0);

        sub_ = this->create_subscription<ur_sim::msg::DetectedObjectArray>(
            "/detected_objects", 10,
            std::bind(&MinimalPickPlace::callback, this, std::placeholders::_1));

        attach_client_ = this->create_client<linkattacher_msgs::srv::AttachLink>("/ATTACHLINK");
        detach_client_ = this->create_client<linkattacher_msgs::srv::DetachLink>("/DETACHLINK");

        RCLCPP_INFO(this->get_logger(), "✅ MINIMAL VERSION READY");
    }

private:

    bool busy = false;

    // 🔥 SAFE MOVE (JOINT BASED)
    bool moveSafe(const geometry_msgs::msg::Pose &pose)
    {
        move_group.setStartStateToCurrentState();

        if (!move_group.setApproximateJointValueTarget(pose))
        {
            RCLCPP_WARN(this->get_logger(), "IK failed");
            return false;
        }

        moveit::planning_interface::MoveGroupInterface::Plan plan;

        if (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
        {
            move_group.execute(plan);
            return true;
        }

        return false;
    }

    void goHome()
    {
        move_group.setNamedTarget("home");
        move_group.move();
    }

    void attachObject(const std::string &name)
    {
        auto req = std::make_shared<linkattacher_msgs::srv::AttachLink::Request>();
        req->model1_name = "cobot";
        req->link1_name = "wrist_3_link";
        req->model2_name = name;
        req->link2_name = "link_1";

        attach_client_->wait_for_service();
        attach_client_->async_send_request(req);
    }

    void detachObject(const std::string &name)
    {
        auto req = std::make_shared<linkattacher_msgs::srv::DetachLink::Request>();
        req->model1_name = "cobot";
        req->link1_name = "wrist_3_link";
        req->model2_name = name;
        req->link2_name = "link_1";

        detach_client_->wait_for_service();
        detach_client_->async_send_request(req);
    }

    void callback(const ur_sim::msg::DetectedObjectArray::SharedPtr msg)
    {
        if (busy || msg->objects.empty())
            return;

        busy = true;

        for (const auto &obj : msg->objects)
        {
            RCLCPP_INFO(this->get_logger(), "Picking %s", obj.label.c_str());

            goHome();
            rclcpp::sleep_for(1s);

            geometry_msgs::msg::Pose target = obj.pose;

            // 🔥 SAFE HEIGHTS
            target.position.z = 0.25;
            target.orientation.w = 1.0;

            geometry_msgs::msg::Pose above = target;
            above.position.z = 0.5;

            // MOVE ABOVE
            if (!moveSafe(above)) continue;
            rclcpp::sleep_for(1s);

            // MOVE DOWN
            if (!moveSafe(target)) continue;
            rclcpp::sleep_for(1s);

            // ATTACH
            attachObject(obj.label);
            rclcpp::sleep_for(1s);

            // MOVE UP
            if (!moveSafe(above)) continue;
            rclcpp::sleep_for(1s);

            // PLACE
            geometry_msgs::msg::Pose place;
            place.position.x = 0.3;
            place.position.y = 0.5;
            place.position.z = 0.5;
            place.orientation.w = 1.0;

            moveSafe(place);
            rclcpp::sleep_for(1s);

            // DETACH
            detachObject(obj.label);
            rclcpp::sleep_for(1s);
        }

        busy = false;
        RCLCPP_INFO(this->get_logger(), "✅ DONE");
    }

    rclcpp::Subscription<ur_sim::msg::DetectedObjectArray>::SharedPtr sub_;
    moveit::planning_interface::MoveGroupInterface move_group;

    rclcpp::Client<linkattacher_msgs::srv::AttachLink>::SharedPtr attach_client_;
    rclcpp::Client<linkattacher_msgs::srv::DetachLink>::SharedPtr detach_client_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MinimalPickPlace>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
