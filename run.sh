#!/usr/bin/bash
sudo modprobe v4l2loopback card_label="Stage Camera" video_nr=2 exclusive_caps=1

if [ $1 == "--setup" ]; then
  echo "Installing dependencies!"
  pip3 install numpy opencv_python pyvirtualcam argparse scipy imutils
  sudo apt install v4l2loopback-dkms
else
  python3 face_tracker.py $@
fi
# Copyright https://github/vega-d 2021