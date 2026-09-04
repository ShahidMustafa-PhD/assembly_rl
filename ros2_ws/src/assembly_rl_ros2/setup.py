from setuptools import find_packages, setup
from glob import glob

package_name = "assembly_rl_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Shahid Mustafa",
    maintainer_email="shahid.pieas@gmail.com",
    description="Perception -> policy -> actuation ROS2 nodes for vision-guided assembly RL.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = assembly_rl_ros2.perception_node:main",
            "policy_node = assembly_rl_ros2.policy_node:main",
            "impedance_controller_node = assembly_rl_ros2.impedance_controller_node:main",
        ],
    },
)
