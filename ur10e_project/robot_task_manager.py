import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


class RobotTaskManager(Node):
    def __init__(self):
        super().__init__("robot_task_manager")

        self.subscription = self.create_subscription(
            String,
            "/human_intention",
            self.human_intention_callback,
            10
        )

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # 当前任务状态
        self.busy = False
        self.last_human_target = None

        # 颜色对应的“安全测试姿态”
        # 注意：这些只是 fake hardware 测试姿态，不要直接用于真机
        self.joint_targets = {
            "red": [
                -0.8,
                -1.6,
                1.5,
                -1.5,
                -1.57,
                0.0,
            ],
            "blue": [
                -0.3,
                -1.6,
                1.5,
                -1.5,
                -1.57,
                0.0,
            ],
            "green": [
                0.3,
                -1.6,
                1.5,
                -1.5,
                -1.57,
                0.0,
            ],
            "yellow": [
                0.8,
                -1.6,
                1.5,
                -1.5,
                -1.57,
                0.0,
            ],
            "home": [
                0.0,
                -1.57,
                1.57,
                -1.57,
                -1.57,
                0.0,
            ],
        }

        self.block_order = ["red", "blue", "green", "yellow"]

        self.get_logger().info("Robot task manager started.")
        self.get_logger().info("Waiting for /human_intention...")

    def choose_robot_target(self, human_target):
        """
        当前策略 A：
        机器人避开人的目标，然后选择预设顺序中第一个可用目标。
        这里先用简单稳定逻辑，后面可以替换成策略 A/B/C。
        """
        for block in self.block_order:
            if block != human_target:
                return block

        return None

    def human_intention_callback(self, msg):
        human_target = msg.data.strip()

        if human_target not in self.block_order:
            self.get_logger().warn(f"Invalid human target: {human_target}")
            return

        if self.busy:
            self.get_logger().warn(
                f"Robot is busy. Ignored new human intention: {human_target}"
            )
            return

        if human_target == self.last_human_target:
            self.get_logger().info(
                f"Repeated human intention ignored: {human_target}"
            )
            return

        self.last_human_target = human_target

        robot_target = self.choose_robot_target(human_target)

        if robot_target is None:
            self.get_logger().error("No valid robot target found.")
            return

        self.get_logger().info(
            f"Human target: {human_target}. Robot avoids it and chooses: {robot_target}"
        )

        self.send_joint_goal(robot_target)

    def send_joint_goal(self, robot_target):
        self.busy = True

        self.get_logger().info("Waiting for UR10e trajectory action server...")
        self.action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        # 第一段：移动到目标颜色对应姿态
        point1 = JointTrajectoryPoint()
        point1.positions = self.joint_targets[robot_target]
        point1.time_from_start.sec = 4
        point1.time_from_start.nanosec = 0

        # 第二段：回到 home 姿态
        point2 = JointTrajectoryPoint()
        point2.positions = self.joint_targets["home"]
        point2.time_from_start.sec = 8
        point2.time_from_start.nanosec = 0

        goal_msg.trajectory.points.append(point1)
        goal_msg.trajectory.points.append(point2)

        self.get_logger().info(
            f"Sending UR10e trajectory for robot target: {robot_target}"
        )

        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            self.busy = False
            return

        self.get_logger().info("Trajectory goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result

        self.get_logger().info(
            f"Trajectory finished. error_code={result.error_code}"
        )

        self.busy = False
        self.get_logger().info("Robot is ready for next intention.")


def main(args=None):
    rclpy.init(args=args)

    node = RobotTaskManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
