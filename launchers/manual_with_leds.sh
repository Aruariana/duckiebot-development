#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launching manual mode with LED supervisor
# This launch file includes the global Duckietown master launch and enables LEDs and object detection.
dt-exec roslaunch --wait duckietown_demos manual_with_leds.launch

# wait for app to end
dt-launchfile-join
