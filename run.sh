#!/bin/bash
source /opt/ros/jazzy/setup.bash

# 1. Launch the ROS Bridge in a NEW, separate terminal window
# The -- tab command tells gnome-terminal to open a new tab/window
gnome-terminal -- bash -c "source /opt/ros/jazzy/setup.bash; ./start_bridge.sh; exec bash"

# 2. Wait for the bridge to initialize
sleep 2

#Block Gazebo from wasting time searching the online Fuel database
export GAZEBO_MODEL_DATABASE_URI=""

export GAZEBO_MODEL_PATH=$(pwd):$GAZEBO_MODEL_PATH

export GZ_SIM_RESOURCE_PATH=$(pwd):$GZ_SIM_RESOURCE_PATH

gz sim ground_tracking.sdf &

