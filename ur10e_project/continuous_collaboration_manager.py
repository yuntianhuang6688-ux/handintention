import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class ContinuousCollaborationManager(Node):
    def __init__(self):
        super().__init__("continuous_collaboration_manager")

        # ==========================
        # ROS2 subscription: human intention
        # ==========================
        self.subscription = self.create_subscription(
            String,
            "/human_intention",
            self.human_intention_callback,
            10
        )

        # ==========================
        # UR10e trajectory action client
        # ==========================
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

        # ==========================
        # MoveIt planning scene publisher
        # ==========================
        self.collision_pub = self.create_publisher(
            CollisionObject,
            "/collision_object",
            10
        )

        # ==========================
        # UDP: WSL -> Windows
        # 用来把剩余方块列表发回 Windows
        # ==========================
        self.windows_ip = "172.26.144.1"   # 改成 cat /etc/resolv.conf 里的 nameserver IP
        self.windows_port = 5006
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ==========================
        # Task settings
        # ==========================
        self.block_order = ["red", "blue", "green", "yellow", "orange"]

        self.blocks = {
            color: {
                "available": True,
                "picked_by": None,
                "placed": False
            }
            for color in self.block_order
        }

        self.stack_level = 0
        self.busy = False
        self.last_processed_human_target = None
        self.current_human_target = None
        self.current_robot_target = None
        self.current_robot_stack_layer = None

        # ==========================
        # Scene geometry
        # 这些坐标是 RViz / fake hardware 测试坐标
        # 后面可以根据真实桌面调整
        # ==========================
        self.block_size = [0.05, 0.05, 0.05]
        self.block_height = 0.05

        self.block_positions = {
            "red":    [0.35, -0.25, 0.05],
            "blue":   [0.45, -0.25, 0.05],
            "green":  [0.55, -0.25, 0.05],
            "yellow": [0.65, -0.25, 0.05],
            "orange": [0.75, -0.25, 0.05],
        }

        # 搭塔位置
        self.stack_base_position = [0.55, 0.20, 0.05]

        # ==========================
        # Carrying visualization state
        # ==========================
        self.carrying_visualization_active = False
        self.carrying_start_time = None
        self.carrying_robot_target = None
        self.carrying_final_position = None

        # 每 0.1 秒更新一次机器人搬运方块的可视化位置
        self.carrying_timer = self.create_timer(
            0.1,
            self.update_carrying_visualization
        )

        # ==========================
        # Fake hardware joint targets
        # 这些只是为了让 RViz 里的 UR10e 动起来
        # 不要直接用于真机
        # ==========================
        self.joint_targets = {
            "home": [
                0.0,
                -1.57,
                1.57,
                -1.57,
                -1.57,
                0.0,
            ],

            # red pick sequence
            "red_above": [
                -0.90,
                -1.45,
                1.35,
                -1.45,
                -1.57,
                0.0,
            ],
            "red_grasp": [
                -0.90,
                -1.65,
                1.55,
                -1.50,
                -1.57,
                0.0,
            ],

            # blue pick sequence
            "blue_above": [
                -0.45,
                -1.45,
                1.35,
                -1.45,
                -1.57,
                0.0,
            ],
            "blue_grasp": [
                -0.45,
                -1.65,
                1.55,
                -1.50,
                -1.57,
                0.0,
            ],

            # green pick sequence
            "green_above": [
                0.0,
                -1.45,
                1.35,
                -1.45,
                -1.57,
                0.0,
            ],
            "green_grasp": [
                0.0,
                -1.65,
                1.55,
                -1.50,
                -1.57,
                0.0,
            ],

            # yellow pick sequence
            "yellow_above": [
                0.45,
                -1.45,
                1.35,
                -1.45,
                -1.57,
                0.0,
            ],
            "yellow_grasp": [
                0.45,
                -1.65,
                1.55,
                -1.50,
                -1.57,
                0.0,
            ],

            # orange pick sequence
            "orange_above": [
                0.90,
                -1.45,
                1.35,
                -1.45,
                -1.57,
                0.0,
            ],
            "orange_grasp": [
                0.90,
                -1.65,
                1.55,
                -1.50,
                -1.57,
                0.0,
            ],

            # stack sequence
            "stack_above": [
                0.20,
                -1.35,
                1.25,
                -1.45,
                -1.57,
                0.0,
            ],
            "stack_place": [
                0.20,
                -1.55,
                1.45,
                -1.50,
                -1.57,
                0.0,
            ],
        }

        self.get_logger().info("Continuous collaboration manager started.")
        self.get_logger().info(
            f"Remaining blocks will be sent to Windows: {self.windows_ip}:{self.windows_port}"
        )

        time.sleep(1.0)

        self.reset_planning_scene()
        self.print_task_state()
        self.send_remaining_blocks_to_windows()

    # =========================================================
    # Utility functions
    # =========================================================
    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def lerp(self, start, end, alpha):
        alpha = max(0.0, min(1.0, alpha))
        return [
            start[0] + alpha * (end[0] - start[0]),
            start[1] + alpha * (end[1] - start[1]),
            start[2] + alpha * (end[2] - start[2]),
        ]

    # =========================================================
    # Planning scene functions
    # =========================================================
    def make_box_object(self, object_id, size, position):
        obj = CollisionObject()
        obj.header.frame_id = "base_link"
        obj.id = object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.w = 1.0

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        return obj

    def publish_object(self, object_id, size, position):
        obj = self.make_box_object(object_id, size, position)
        self.collision_pub.publish(obj)

    def publish_object_with_log(self, object_id, size, position):
        self.publish_object(object_id, size, position)
        self.get_logger().info(
            f"Scene object updated: {object_id}, position={position}"
        )
        time.sleep(0.15)

    def remove_object(self, object_id):
        obj = CollisionObject()
        obj.header.frame_id = "base_link"
        obj.id = object_id
        obj.operation = CollisionObject.REMOVE

        self.collision_pub.publish(obj)
        self.get_logger().info(f"Scene object removed: {object_id}")
        time.sleep(0.15)

    def reset_planning_scene(self):
        """
        重新添加桌子、5 个方块、搭塔区域。
        """
        self.get_logger().info("Resetting planning scene...")

        object_ids = [
            "table",
            "stack_target",
            "red_block",
            "blue_block",
            "green_block",
            "yellow_block",
            "orange_block",
        ]

        for object_id in object_ids:
            self.remove_object(object_id)

        time.sleep(0.5)

        # 添加桌子
        table_size = [1.2, 0.8, 0.05]
        table_pos = [0.45, 0.0, -0.03]
        self.publish_object_with_log("table", table_size, table_pos)

        # 添加初始方块
        for color in self.block_order:
            self.publish_object_with_log(
                f"{color}_block",
                self.block_size,
                self.block_positions[color]
            )

        # 添加搭塔目标区域
        target_size = [0.16, 0.16, 0.01]
        target_pos = [
            self.stack_base_position[0],
            self.stack_base_position[1],
            0.005
        ]
        self.publish_object_with_log("stack_target", target_size, target_pos)

        self.get_logger().info("Planning scene reset completed.")

    def get_stack_position(self, layer_index):
        x = self.stack_base_position[0]
        y = self.stack_base_position[1]
        z = self.stack_base_position[2] + layer_index * self.block_height
        return [x, y, z]

    def move_block_to_stack(self, color, layer_index):
        """
        把某个颜色的方块移动到塔上。
        """
        object_id = f"{color}_block"
        new_position = self.get_stack_position(layer_index)

        self.publish_object_with_log(
            object_id,
            self.block_size,
            new_position
        )

        self.get_logger().info(
            f"{object_id} moved to stack layer {layer_index + 1}"
        )

    # =========================================================
    # Carrying visualization
    # =========================================================
    def start_robot_block_carrying_visualization(self, robot_target, final_layer):
        """
        启动机器人搬运目标方块的 RViz 可视化。
        方块会在机器人 fake trajectory 执行过程中逐步移动。
        """
        self.carrying_visualization_active = True
        self.carrying_start_time = self.now_sec()
        self.carrying_robot_target = robot_target
        self.carrying_final_position = self.get_stack_position(final_layer)

        self.get_logger().info(
            f"Started carrying visualization for {robot_target}_block "
            f"to layer {final_layer + 1}"
        )

    def update_carrying_visualization(self):
        """
        根据 fake pick-and-place 时间轴移动 robot_target 的方块。
        时间轴与 joint trajectory 保持一致：

        0-4 s: 方块留在桌面
        4-6 s: 方块被抬起
        6-8 s: 方块移动到塔上方
        8-10 s: 方块下降到目标层
        10-14 s: 方块停在目标层
        """
        if not self.carrying_visualization_active:
            return

        if self.carrying_robot_target is None:
            return

        elapsed = self.now_sec() - self.carrying_start_time
        color = self.carrying_robot_target
        object_id = f"{color}_block"

        start_pos = self.block_positions[color]
        lifted_pos = [
            start_pos[0],
            start_pos[1],
            start_pos[2] + 0.20
        ]

        final_pos = self.carrying_final_position
        above_stack_pos = [
            final_pos[0],
            final_pos[1],
            final_pos[2] + 0.20
        ]

        # 0-4s：还在桌面
        if elapsed < 4.0:
            pos = start_pos

        # 4-6s：从桌面抬起
        elif elapsed < 6.0:
            alpha = (elapsed - 4.0) / 2.0
            pos = self.lerp(start_pos, lifted_pos, alpha)

        # 6-8s：移动到塔上方
        elif elapsed < 8.0:
            alpha = (elapsed - 6.0) / 2.0
            pos = self.lerp(lifted_pos, above_stack_pos, alpha)

        # 8-10s：下降到塔上
        elif elapsed < 10.0:
            alpha = (elapsed - 8.0) / 2.0
            pos = self.lerp(above_stack_pos, final_pos, alpha)

        # 10s 之后：保持在塔上
        else:
            pos = final_pos

        self.publish_object(
            object_id,
            self.block_size,
            pos
        )

        if elapsed >= 14.0:
            self.carrying_visualization_active = False
            self.publish_object_with_log(
                object_id,
                self.block_size,
                final_pos
            )
            self.get_logger().info(
                f"Carrying visualization finished for {object_id}"
            )

    # =========================================================
    # Task state functions
    # =========================================================
    def print_task_state(self):
        self.get_logger().info("========== Current Task State ==========")
        for color, state in self.blocks.items():
            self.get_logger().info(
                f"{color:6s} | available={state['available']} | "
                f"picked_by={state['picked_by']} | placed={state['placed']}"
            )
        self.get_logger().info(f"Current stack level: {self.stack_level}")
        self.get_logger().info("========================================")

    def get_remaining_blocks(self):
        return [
            color for color in self.block_order
            if self.blocks[color]["available"]
        ]

    def send_remaining_blocks_to_windows(self):
        remaining = self.get_remaining_blocks()
        message = "remaining:" + ",".join(remaining)

        try:
            self.udp_sock.sendto(
                message.encode("utf-8"),
                (self.windows_ip, self.windows_port)
            )
            self.get_logger().info(f"Sent to Windows: {message}")
        except Exception as e:
            self.get_logger().error(
                f"Failed to send remaining blocks to Windows: {e}"
            )

    def choose_robot_target(self, human_target):
        """
        当前简单策略：
        1. 不选择人类目标
        2. 不选择已经被拿走/放置的积木
        3. 从剩余 available 积木中按固定顺序选择一个

        后续可以替换成：
        - nearest-to-robot
        - farthest-from-human
        - safety-efficiency weighted strategy
        """
        for color in self.block_order:
            if color == human_target:
                continue

            if self.blocks[color]["available"]:
                return color

        return None

    # =========================================================
    # ROS callback
    # =========================================================
    def human_intention_callback(self, msg):
        human_target = msg.data.strip()

        if human_target not in self.block_order:
            self.get_logger().warn(
                f"Invalid human target received: {human_target}"
            )
            return

        if self.busy:
            self.get_logger().warn(
                f"Robot is busy. Ignored human intention: {human_target}"
            )
            return

        if human_target == self.last_processed_human_target:
            self.get_logger().info(
                f"Repeated human target ignored: {human_target}"
            )
            return

        if not self.blocks[human_target]["available"]:
            self.get_logger().warn(
                f"Human target {human_target} is already unavailable. Ignored."
            )
            self.send_remaining_blocks_to_windows()
            return

        self.last_processed_human_target = human_target
        self.current_human_target = human_target

        self.get_logger().info("")
        self.get_logger().info(
            f"New round: human intends to pick {human_target}"
        )

        # ==========================
        # 只记录人类目标，不立刻移动到塔上
        # 人类方块会在机器人动作完成后一起更新到塔上
        # ==========================
        self.blocks[human_target]["available"] = False
        self.blocks[human_target]["picked_by"] = "human"
        self.blocks[human_target]["placed"] = False

        self.get_logger().info(
            f"Human block {human_target} is reserved, but not placed yet."
        )

        # ==========================
        # 机器人选择另一个积木
        # ==========================
        robot_target = self.choose_robot_target(human_target)

        if robot_target is None:
            self.get_logger().info("No available block left for robot.")

            human_layer = self.stack_level
            self.blocks[human_target]["placed"] = True
            self.move_block_to_stack(human_target, human_layer)
            self.stack_level += 1

            self.print_task_state()
            self.send_remaining_blocks_to_windows()
            return

        self.blocks[robot_target]["available"] = False
        self.blocks[robot_target]["picked_by"] = "robot"
        self.blocks[robot_target]["placed"] = False
        self.current_robot_target = robot_target

        # 机器人方块最终会放在人类方块的上面
        # 所以 robot layer = 当前 stack_level + 1
        self.current_robot_stack_layer = self.stack_level + 1

        self.get_logger().info(
            f"Robot avoids {human_target} and chooses {robot_target}."
        )

        self.send_robot_sequence(robot_target)

    # =========================================================
    # UR10e fake multi-step pick-and-place motion
    # =========================================================
    def send_robot_sequence(self, robot_target):
        self.busy = True

        self.get_logger().info("Waiting for UR10e trajectory action server...")
        self.action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        trajectory_steps = [
            (
                "move above selected block",
                self.joint_targets[f"{robot_target}_above"],
                2
            ),
            (
                "move down to grasp selected block",
                self.joint_targets[f"{robot_target}_grasp"],
                4
            ),
            (
                "lift selected block",
                self.joint_targets[f"{robot_target}_above"],
                6
            ),
            (
                "move above stack",
                self.joint_targets["stack_above"],
                8
            ),
            (
                "move down to place block",
                self.joint_targets["stack_place"],
                10
            ),
            (
                "return above stack",
                self.joint_targets["stack_above"],
                12
            ),
            (
                "return home",
                self.joint_targets["home"],
                14
            ),
        ]

        self.get_logger().info(
            f"Starting fake pick-and-place sequence for robot target: {robot_target}"
        )

        for step_name, joint_position, time_sec in trajectory_steps:
            point = JointTrajectoryPoint()
            point.positions = joint_position
            point.time_from_start.sec = time_sec
            point.time_from_start.nanosec = 0
            goal_msg.trajectory.points.append(point)

            self.get_logger().info(
                f"Queued step at {time_sec}s: {step_name}"
            )

        # 启动方块跟随搬运路径的可视化
        if self.current_robot_stack_layer is not None:
            self.start_robot_block_carrying_visualization(
                robot_target,
                self.current_robot_stack_layer
            )

        future = self.action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            self.busy = False
            self.carrying_visualization_active = False
            return

        self.get_logger().info("Trajectory goal accepted.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result

        human_target = self.current_human_target
        robot_target = self.current_robot_target

        self.get_logger().info(
            f"Trajectory finished for {robot_target}. "
            f"error_code={result.error_code}"
        )

        # 停止搬运可视化，确保机器人方块最终在正确层
        self.carrying_visualization_active = False

        # ==========================
        # 机器人动作完成后，再把人类方块移动到塔上
        # ==========================
        if human_target is not None and not self.blocks[human_target]["placed"]:
            human_layer = self.stack_level

            self.blocks[human_target]["placed"] = True
            self.move_block_to_stack(human_target, human_layer)

            self.stack_level += 1

            self.get_logger().info(
                f"Human block {human_target} is now recorded as placed. "
                f"Stack level is now {self.stack_level}."
            )

        # ==========================
        # 再把机器人方块移动到塔上
        # ==========================
        if robot_target is not None and not self.blocks[robot_target]["placed"]:
            robot_layer = self.stack_level

            self.blocks[robot_target]["placed"] = True
            self.move_block_to_stack(robot_target, robot_layer)

            self.stack_level += 1

            self.get_logger().info(
                f"Robot block {robot_target} is now recorded as placed. "
                f"Stack level is now {self.stack_level}."
            )

        self.busy = False

        self.print_task_state()
        self.send_remaining_blocks_to_windows()

        remaining = self.get_remaining_blocks()

        if len(remaining) == 0:
            self.get_logger().info(
                "All blocks have been used. Collaboration task finished."
            )
        else:
            self.get_logger().info(
                f"Remaining blocks: {remaining}. "
                f"Waiting for next human intention."
            )

        # 清理当前轮任务变量
        self.current_human_target = None
        self.current_robot_target = None
        self.current_robot_stack_layer = None
        self.carrying_robot_target = None
        self.carrying_final_position = None

    def destroy_node(self):
        self.udp_sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = ContinuousCollaborationManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()