#!/bin/bash

# fill in missing frames

# Directory containing the frames
FRAMES_DIR="visualization_frames"
# Change to the frames directory
cd "$FRAMES_DIR" || exit
# Get the list of existing frames sorted by their number
# existing_frames=($(ls frame_*.png | sort -V))
existing_frames=($(ls *.png | sort -V))
# Iterate through the existing frames
for frame in "${existing_frames[@]}"; do
    # Extract the frame number from the filename
    frame_number=$(echo "$frame" | grep -oE '[0-9]+')
    # Check for the next frame
    next_frame_number=$((frame_number + 1))
    next_frame="frame_${next_frame_number}.png"
    # If the next frame does not exist, copy the current frame to fill the gap
    if [[ ! -f "$next_frame" ]]; then
        echo "Missing $next_frame. Filling it with $frame."
        cp "$frame" "$next_frame"
    fi
done
echo "Frame filling complete."

# make a movie
cd ../
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libffi.so.7
ffmpeg -framerate 10 -i visualization_frames/frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4 -y
