export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libffi.so.7
ffmpeg -framerate 10 -i frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4