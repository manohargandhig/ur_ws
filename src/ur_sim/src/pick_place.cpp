#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <memory>
#include <vector>

class SequentialPickPlace : public rclcpp::Node
{
public:
    SequentialPickPlace() : Node("sequential_pick_place"), place_index_(0)
    {
        sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/detected_objects", 10,
            std::bind(&SequentialPickPlace::callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Waiting for detected objects...");
    }

private:
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    int place_index_;

    // ---------------- INIT MOVEIT ----------------
    void init_moveit()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "ur5_manipulator");

        move_group_->setPlanningTime(10.0);
        move_group_->setMaxVelocityScalingFactor(0.3);
        move_group_->setMaxAccelerationScalingFactor(0.3);

        addTable();

        RCLCPP_INFO(this->get_logger(), "MoveIt initialized");
    }

    // ---------------- ADD TABLE ----------------
    void addTable()
    {
        moveit_msgs::msg::CollisionObject table;
        table.header.frame_id = "base_link";
        table.id = "table";

        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions = {1.0, 1.0, 0.05};

        geometry_msgs::msg::Pose pose;
        pose.orientation.w = 1.0;

        pose.position.x = 0.6;
        pose.position.y = 0.0;
        pose.position.z = 0.35;

        table.primitives.push_back(primitive);
        table.primitive_poses.push_back(pose);
        table.operation = table.ADD;

        planning_scene_interface_.applyCollisionObject(table);

        RCLCPP_INFO(this->get_logger(), "Table added");
    }

    // ---------------- MAIN CALLBACK ----------------
    void callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        if (!move_group_)
        {
            init_moveit();
        }

        if (msg->poses.empty())
            return;

        RCLCPP_INFO(this->get_logger(), "Received %ld objects", msg->poses.size());

        for (size_t i = 0; i < msg->poses.size(); ++i)
        {
            auto target = msg->poses[i];

            // ---------- WORKSPACE FILTER ----------
            if (target.position.x > 0.8 || target.position.x < 0.1 ||
                target.position.y > 0.5 || target.position.y < -0.5)
            {
                RCLCPP_WARN(this->get_logger(), "Skipping unreachable object");
                continue;
            }

            // ---------- FIX Z (TABLE HEIGHT) ----------
            target.position.z = 0.78;

            // ---------- FIX ORIENTATION ----------
            target.orientation.x = 0.707;
            target.orientation.y = 0.0;
            target.orientation.z = 0.707;
            target.orientation.w = 0.0;

            RCLCPP_INFO(this->get_logger(),
                        "Picking object %ld at (%.3f, %.3f, %.3f)",
                        i,
                        target.position.x,
                        target.position.y,
                        target.position.z);

            // ---------- PRE-GRASP ----------
            geometry_msgs::msg::Pose pre = target;
            pre.position.z = 0.95;

            move_group_->setPoseTarget(pre);
            if (move_group_->move() != moveit::core::MoveItErrorCode::SUCCESS)
            {
                RCLCPP_WARN(this->get_logger(), "Pre-grasp failed");
                continue;
            }

            // ---------- APPROACH ----------
            move_group_->setPoseTarget(target);
            if (move_group_->move() != moveit::core::MoveItErrorCode::SUCCESS)
            {
                RCLCPP_WARN(this->get_logger(), "Approach failed");
                continue;
            }

            // 👉 GRIPPER CLOSE (optional)

            // ---------- LIFT ----------
            move_group_->setPoseTarget(pre);
            move_group_->move();

            // ===============================
            // 📦 PLACE IN BOX
            // ===============================

            double box_x = 0.75;
            double box_y = 0.0;
            double box_z = 0.70;

            int row = place_index_ / 3;
            int col = place_index_ % 3;

            double offset_x = -0.05 + col * 0.05;
            double offset_y = -0.05 + row * 0.05;

            geometry_msgs::msg::Pose place;
            place.position.x = box_x + offset_x;
            place.position.y = box_y + offset_y;
            place.position.z = box_z;

            place.orientation = target.orientation;

            place_index_++;

            // ---------- ABOVE BOX ----------
            geometry_msgs::msg::Pose place_pre = place;
            place_pre.position.z = 0.95;

            move_group_->setPoseTarget(place_pre);
            move_group_->move();

            // ---------- LOWER ----------
            move_group_->setPoseTarget(place);
            move_group_->move();

            // 👉 GRIPPER OPEN (optional)

            // ---------- RETREAT ----------
            move_group_->setPoseTarget(place_pre);
            move_group_->move();

            RCLCPP_INFO(this->get_logger(), "Placed object %ld", i);
        }

        RCLCPP_INFO(this->get_logger(), "Finished all objects");
    }
};

// ---------------- MAIN ----------------
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<SequentialPickPlace>();

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
