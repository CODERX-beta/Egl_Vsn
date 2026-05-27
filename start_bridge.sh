#!/bin/bash


source /opt/ros/jazzy/setup.bash

ros2 run ros_gz_bridge parameter_bridge /model/parrot_bebop_2/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist /world/empty/model/parrot_bebop_2/link/front_camera_link/sensor/front_camera/image@sensor_msgs/msg/Image@gz.msgs.Image

