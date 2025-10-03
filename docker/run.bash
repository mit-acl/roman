#!/usr/bin/env bash
# ROS_VARS="-e ROS_MASTER_URI=$ROS_MASTER_URI -e ROS_IP=$ROS_IP -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
# KIMERA_VARS="-e CAMERA=$KIMERA_VIO_CAMERA -e ROBOT_NAME=$ROBOT_NAME"

VOLUMES="-v $ROMAN_DEMO_DATA:/roman_demo_data"
VARS="-e ROMAN_DEMO_DATA=/roman_demo_data -e ROMAN_WEIGHTS=/roman/weights"

docker run -it --rm --net=host \
    $VOLUMES $VARS --gpus all roman
