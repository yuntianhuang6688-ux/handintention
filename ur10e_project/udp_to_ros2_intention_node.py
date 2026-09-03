import socket
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class UDPToROS2IntentionNode(Node):
    def __init__(self):
        super().__init__("udp_to_ros2_intention_node")

        self.publisher = self.create_publisher(
            String,
            "/human_intention",
            10
        )

        self.udp_ip = "0.0.0.0"
        self.udp_port = 5005

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))

        self.get_logger().info(
            f"Listening for UDP human intention on port {self.udp_port}"
        )
        self.get_logger().info(
            "Publishing received intention to ROS2 topic: /human_intention"
        )

        self.running = True

        self.thread = threading.Thread(target=self.udp_receive_loop)
        self.thread.daemon = True
        self.thread.start()

    def udp_receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                intention = data.decode("utf-8").strip()

                if intention not in ["red", "blue", "green", "yellow", "orange"]:
                    self.get_logger().warn(
                        f"Ignored invalid intention from {addr}: {intention}"
                    )
                    continue

                msg = String()
                msg.data = intention
                self.publisher.publish(msg)

                self.get_logger().info(
                    f"Received UDP: {intention} from {addr} -> published /human_intention"
                )

            except Exception as e:
                self.get_logger().error(f"UDP receive error: {e}")

    def destroy_node(self):
        self.running = False
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = UDPToROS2IntentionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
