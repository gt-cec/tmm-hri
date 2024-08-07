# preprocess_frames.py: runs all frames through the semantic segmentation network so we can develop faster, actual robots should use real-time segmentation

import pickle
import cv2
import os
from segmentation import scene_segmentation

if __name__ == "__main__":
    sim_dir = "../Output/human/0"
    files = os.listdir(sim_dir)
    frames = sorted([int(x.split("_")[1]) for x in files if x.startswith("Action") and x.endswith(".png")])  # get frames, the .png filter prevents duplicates (each frame has .png and .exr)
    for frame_id in frames:
        seg_file_name = "seg_" + str(frame_id) + "_detectron2.pkl"
        if seg_file_name in files:
            print("Skipping", frame_id, "as already pickled.")
            continue
        print("Opening frame", frame_id)
        bgr = cv2.imread(sim_dir + "/Action_" + str(frame_id).zfill(4) + "_0_normal.png")  # read the image, OpenCV defaults to BGR
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
        print("   Segmenting...")
        labels, seg_masks = scene_segmentation.segmentation(rgb)  # segment
        print("   Saving...")
        pickle.dump([labels, seg_masks], open(sim_dir + "/" + seg_file_name, "wb"))  # save

    print("All done!")
