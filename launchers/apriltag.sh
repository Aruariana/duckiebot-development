#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launching apriltag combined launcher (detector + localization)
dt-exec roslaunch --wait duckietown_demos apriltag.launch

# wait for app to end
dt-launchfile-join
