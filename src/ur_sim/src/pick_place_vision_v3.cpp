#include <memory>
#include <chrono>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <std_msgs/msg/string.hpp>

#include <linkattacher_msgs/srv/attach_link.hpp>
#include <linkattacher_msgs/srv/detach_link.hpp>

using namespace std::chrono_literals;

/**
 * VisionPickPlace — updated to work with multi_object_perception_node.py
 *
 * Listens to /perception/current_target for the object name,
 * then moves to object_pregrasp_frame → object_grasp_frame,
 * attaches via IFRA LinkAttacher, places, detaches, loops.
 *
 * All frame names come from the perception node — zero hardcoding.
 */
class VisionPickPlace : public rclcpp::Node
{
public:
    VisionPickPlace()
        : Node("vision_pick_place_node"),
          move_group_(std::shared_ptr<rclcpp::Node>(this), "ur5_manipulator"),
          gripper_(std::shared_ptr<rclcpp::Node>(this), "robotiq_gripper"),
          current_object_name_(""),
          is_executing_(false)
    {
        // Reference frame must be base_link (matches perception node output)
        move_group_.setPoseReferenceFrame("base_link");
        move_group_.setPlanningTime(10.0);
        move_group_.setMaxVelocityScalingFactor(0.3);
        move_group_.setMaxAccelerationScalingFactor(0.3);

        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        attach_client_ = this->create_client<linkattacher_msgs::srv::AttachLink>("/ATTACHLINK");
        detach_client_ = this->create_client<linkattacher_msgs::srv::DetachLink>("/DETACHLINK");

        // Subscribe to perception node's current target
        target_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/perception/current_target", 10,
            [this](const std_msgs::msg::String::SharedPtr msg)
            {
                if (!is_executing_ && msg->data != current_object_name_)
                {
                    current_object_name_ = msg->data;
                    RCLCPP_INFO(this->get_logger(),
                        "New target detected: %s", current_object_name_.c_str());
                }
            });

        // Place position (fixed drop zone)
        place_pose_.position.x = 0.35;
        place_pose_.position.y = 0.55;
        place_pose_.position.z = 0.50;
        place_pose_.orientation.w = 1.0;

        RCLCPP_INFO(this->get_logger(),
            "VisionPickPlace ready. Waiting for perception node...");

        rclcpp::sleep_for(3s);  // allow TF to populate

        // Main loop timer
        exec_timer_ = this->create_wall_timer(
            500ms, std::bind(&VisionPickPlace::tryExecute, this));
    }

private:

    // ─────────────────────────────────────────────────────────────────
    void tryExecute()
    {
        if (is_executing_ || current_object_name_.empty())
            return;

        is_executing_ = true;
        exec_timer_->cancel();  // pause timer during execution

        RCLCPP_INFO(this->get_logger(),
            "=== Starting pick-place for: %s ===", current_object_name_.c_str());

        bool ok = execute(current_object_name_);

        if (ok)
            RCLCPP_INFO(this->get_logger(), "✅ Pick-place complete: %s",
                current_object_name_.c_str());
        else
            RCLCPP_WARN(this->get_logger(), "⚠️  Pick-place FAILED: %s",
                current_object_name_.c_str());

        current_object_name_ = "";
        is_executing_ = false;

        // Re-enable timer to wait for next object
        exec_timer_ = this->create_wall_timer(
            500ms, std::bind(&VisionPickPlace::tryExecute, this));
    }

    // ─────────────────────────────────────────────────────────────────
    bool execute(const std::string & object_name)
    {
        // TF frame names published by multi_object_perception_node
        const std::string pregrasp_frame = "object_pregrasp_frame";
        const std::string grasp_frame    = "object_grasp_frame";

        // 1. Open gripper
        RCLCPP_INFO(get_logger(), "[1/8] Opening gripper");
        openGripper();
        rclcpp::sleep_for(1s);

        // 2. Move to home pose for safety
        RCLCPP_INFO(get_logger(), "[2/8] Moving to home");
        if (!moveToNamedPose("home")) return false;
        rclcpp::sleep_for(500ms);

        // 3. Move to pre-grasp
        RCLCPP_INFO(get_logger(), "[3/8] Moving to pre-grasp");
        if (!moveToTF(pregrasp_frame)) return false;
        rclcpp::sleep_for(500ms);

        // 4. Move to grasp
        RCLCPP_INFO(get_logger(), "[4/8] Moving to grasp");
        if (!moveToTF(grasp_frame)) return false;
        rclcpp::sleep_for(500ms);

        // 5. Close gripper
        RCLCPP_INFO(get_logger(), "[5/8] Closing gripper");
        closeGripper();
        rclcpp::sleep_for(1s);

        // 6. Attach object in Gazebo physics
        RCLCPP_INFO(get_logger(), "[6/8] Attaching: %s", object_name.c_str());
        attachObject(object_name);
        rclcpp::sleep_for(500ms);

        // 7. Lift to pre-grasp
        RCLCPP_INFO(get_logger(), "[7/8] Lifting to pre-grasp");
        if (!moveToTF(pregrasp_frame)) return false;
        rclcpp::sleep_for(500ms);

        // 8. Move to place position
        RCLCPP_INFO(get_logger(), "[8/8] Moving to place");
        move_group_.setPoseTarget(place_pose_);
        auto plan_result = move_group_.plan(plan_);
        if (plan_result != moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_ERROR(get_logger(), "Place planning failed");
            return false;
        }
        move_group_.execute(plan_);
        rclcpp::sleep_for(500ms);

        // 9. Release
        RCLCPP_INFO(get_logger(), "[9/9] Releasing");
        openGripper();
        detachObject(object_name);
        rclcpp::sleep_for(500ms);

        // 10. Return home
        moveToNamedPose("home");

        return true;
    }

    // ─────────────────────────────────────────────────────────────────
    void openGripper()
    {
        gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.0);
        gripper_.move();
    }

    void closeGripper()
    {
        // 0.79 = fully closed (from SRDF), use 0.60 for safer grasp
        gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.60);
        gripper_.move();
    }

    // ─────────────────────────────────────────────────────────────────
    bool moveToNamedPose(const std::string & pose_name)
    {
        move_group_.setNamedTarget(pose_name);
        auto result = move_group_.plan(plan_);
        if (result != moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_ERROR(get_logger(), "Plan to '%s' failed", pose_name.c_str());
            return false;
        }
        move_group_.execute(plan_);
        return true;
    }

    // ─────────────────────────────────────────────────────────────────
    bool moveToTF(const std::string & frame)
    {
        // Wait up to 3 seconds for the frame to be available
        if (!tf_buffer_->canTransform("base_link", frame,
                                       tf2::TimePointZero,
                                       tf2::durationFromSec(3.0)))
        {
            RCLCPP_ERROR(get_logger(),
                "TF frame '%s' not available", frame.c_str());
            return false;
        }

        try
        {
            auto transform = tf_buffer_->lookupTransform(
                "base_link", frame, tf2::TimePointZero);

            geometry_msgs::msg::Pose pose;
            pose.position.x = transform.transform.translation.x;
            pose.position.y = transform.transform.translation.y;
            pose.position.z = transform.transform.translation.z;
            pose.orientation = transform.transform.rotation;

            move_group_.setPoseTarget(pose);

            auto result = move_group_.plan(plan_);
            if (result != moveit::core::MoveItErrorCode::SUCCESS)
            {
                RCLCPP_ERROR(get_logger(),
                    "Planning to TF frame '%s' failed", frame.c_str());
                return false;
            }
            move_group_.execute(plan_);
            return true;
        }
        catch (const tf2::TransformException & ex)
        {
            RCLCPP_ERROR(get_logger(), "TF error: %s", ex.what());
            return false;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    void attachObject(const std::string & model_name)
    {
        auto req = std::make_shared<linkattacher_msgs::srv::AttachLink::Request>();
        req->model1_name = "cobot";
        req->link1_name  = "wrist_3_link";
        req->model2_name = model_name;       // dynamic — from perception
        req->link2_name  = "link_1";
        attach_client_->wait_for_service();
        attach_client_->async_send_request(req);
    }

    void detachObject(const std::string & model_name)
    {
        auto req = std::make_shared<linkattacher_msgs::srv::DetachLink::Request>();
        req->model1_name = "cobot";
        req->link1_name  = "wrist_3_link";
        req->model2_name = model_name;
        req->link2_name  = "link_1";
        detach_client_->wait_for_service();
        detach_client_->async_send_request(req);
    }

    // ─────────────────────────────────────────────────────────────────
    moveit::planning_interface::MoveGroupInterface move_group_;
    moveit::planning_interface::MoveGroupInterface gripper_;
    moveit::planning_interface::MoveGroupInterface::Plan plan_;

    std::shared_ptr<tf2_ros::Buffer>              tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>   tf_listener_;
    rclcpp::Client<linkattacher_msgs::srv::AttachLink>::SharedPtr attach_client_;
    rclcpp::Client<linkattacher_msgs::srv::DetachLink>::SharedPtr detach_client_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr target_sub_;
    rclcpp::TimerBase::SharedPtr exec_timer_;

    geometry_msgs::msg::Pose place_pose_;
    std::string current_object_name_;
    bool is_executing_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionPickPlace>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
