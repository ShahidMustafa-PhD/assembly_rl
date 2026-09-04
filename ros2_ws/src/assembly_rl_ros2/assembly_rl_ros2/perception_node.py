"""
perception_node -- first stage of the perception -> policy -> actuation loop.

Subscribes to the wrist-mounted eye-in-hand camera's raw image stream,
resizes/center-crops it to match the exact preprocessing AssemblyEnv applies
in simulation (84x84 RGB, see envs/assembly_env.py), and republishes it at a
fixed rate for policy_node. Keeping this resize/crop step identical between
sim and real is what makes the trained CNN's input distribution match --
this is the most common silent sim-to-real bug in vision-RL deployments, so
it is centralized here rather than duplicated in policy_node.

Real camera topic name / resolution depend on hardware (a wrist-mounted
RealSense D405/D435 is typical for UR5e); the constant below is the only
thing that should need changing to swap cameras.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

RAW_IMAGE_TOPIC = "/camera/wrist/image_raw"     # set to your driver's actual topic
PROCESSED_IMAGE_TOPIC = "/assembly/obs/image"
TARGET_SIZE = 84  # must match envs/assembly_env.py's `image_size`


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.bridge = CvBridge()
        self.declare_parameter("target_size", TARGET_SIZE)
        self.target_size = self.get_parameter("target_size").value

        self.sub = self.create_subscription(
            Image, RAW_IMAGE_TOPIC, self._on_image, QoSPresetProfiles.SENSOR_DATA.value
        )
        self.pub = self.create_publisher(Image, PROCESSED_IMAGE_TOPIC, 10)
        self.get_logger().info(
            f"perception_node: {RAW_IMAGE_TOPIC} -> resize({self.target_size}) -> {PROCESSED_IMAGE_TOPIC}"
        )

    def _on_image(self, msg: Image) -> None:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        h, w = cv_img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        cropped = cv_img[y0:y0 + side, x0:x0 + side]
        resized = cv2.resize(cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        out = self.bridge.cv2_to_imgmsg(resized, encoding="rgb8")
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
