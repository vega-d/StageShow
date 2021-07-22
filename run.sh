#!/usr/bin/bash


if [ $1 == "--setup" ]; then
  echo "[INFO] Installing dependencies for StageShow!"
  pip3 install numpy opencv_python pyvirtualcam argparse scipy imutils
  sudo apt install v4l2loopback-dkms v4l2loopback-utils
  echo "[Warning] You might have to reboot now!"
else
  echo "[INFO] Loading in virtual webcam drivers..."
  sudo modprobe -r v4l2loopback
  sudo modprobe v4l2loopback exclusive_caps=1
  echo "[INFO] Loaded in virtual webcam drivers!"
  python3 face_tracker.py $@
fi
# Copyright https://github/vega-d 2021
