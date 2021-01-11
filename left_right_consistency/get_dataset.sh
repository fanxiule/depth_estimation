#!/bin/sh

mkdir dataset
cd dataset
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip
unzip -q data_stereo_flow.zip
rm data_stereo_flow.zip
