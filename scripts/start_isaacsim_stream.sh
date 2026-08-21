#!/bin/bash

source .venv/bin/activate
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$isaac_sim_package_path/exts/isaacsim.ros2.bridge/jazzy/lib

# run isaacsim
isaacsim isaacsim.exp.full.streaming --no-window
