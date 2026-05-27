#!/bin/bash
source /opt/ros/jazzy/setup.bash

#Block Gazebo from wasting time searching the online Fuel database
export GAZEBO_MODEL_DATABASE_URI=""

export GAZEBO_MODEL_PATH=$(pwd):$GAZEBO_MODEL_PATH

export GZ_SIM_RESOURCE_PATH=$(pwd):$GZ_SIM_RESOURCE_PATH

gz sim ground_tracking.sdf &

