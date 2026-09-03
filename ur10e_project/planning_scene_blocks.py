import time

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class PlanningSceneBlocks(Node):
    def __init__(self):
        super().__init__("planning_scene_blocks")

        self.publisher = self.create_publisher(
            CollisionObject,
            "/collision_object",
            10
        )

        self.get_logger().info("Planning scene blocks node started.")

        # 等待 publisher 建立
        time.sleep(1.0)

        self.add_table()
        self.add_blocks()
        self.add_stack_target()

        self.get_logger().info("Table, blocks and stack target have been added.")

    def publish_collision_object(self, object_id, size, position, rgba_name=None):
        """
        添加一个 box collision object 到 MoveIt planning scene.
        size: [x, y, z]
        position: [x, y, z]
        """
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

        self.publisher.publish(obj)

        self.get_logger().info(
            f"Added {object_id}: size={size}, position={position}"
        )

        time.sleep(0.2)

    def add_table(self):
        """
        添加桌子。
        注意：这些坐标只是 fake hardware/RViz 测试坐标。
        之后要根据真实平台调整。
        """
        table_size = [1.2, 0.8, 0.05]
        table_pos = [0.45, 0.0, -0.03]

        self.publish_collision_object(
            "table",
            table_size,
            table_pos
        )

    def add_blocks(self):
        """
        添加 5 个积木。
        方块边长设为 0.05 m。
        """
        block_size = [0.05, 0.05, 0.05]
        block_z = 0.05

        block_positions = {
            "red_block": [0.35, -0.25, block_z],
            "blue_block": [0.45, -0.25, block_z],
            "green_block": [0.55, -0.25, block_z],
            "yellow_block": [0.65, -0.25, block_z],
            "orange_block": [0.75, -0.25, block_z],
        }

        for name, pos in block_positions.items():
            self.publish_collision_object(
                name,
                block_size,
                pos
            )

    def add_stack_target(self):
        """
        添加搭塔目标区域，用一个薄方块表示。
        """
        target_size = [0.15, 0.15, 0.01]
        target_pos = [0.55, 0.20, 0.005]

        self.publish_collision_object(
            "stack_target",
            target_size,
            target_pos
        )


def main(args=None):
    rclpy.init(args=args)

    node = PlanningSceneBlocks()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
