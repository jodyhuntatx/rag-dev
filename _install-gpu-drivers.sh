#!/usr/bin/bash

echo "Should only be needed if Nvidia drivers are not installed/active."
echo "Testing with nvidia-smi:"
nvidia-smi
echo "If test failed, edit script, delete next line and re-run script."
exit -1

sudo apt-get install linux-headers-$(uname -r)
sudo apt -y install ubuntu-drivers-common
sudo ubuntu-drivers install
echo "Rebooting in 5 seconds..."
sleep 5
sudo reboot