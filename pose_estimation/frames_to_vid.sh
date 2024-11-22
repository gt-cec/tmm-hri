#!/bin/bash


# make a movie
ffmpeg -framerate 10 -i frames/top_down_comb_%d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 30 -pix_fmt yuv420p posetest.mp4 -y 
