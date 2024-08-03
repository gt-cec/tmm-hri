# dsg_sim.py: construct a DSG from a simulator

import mental_model
import os, glob
import cv2

sim_dir = "sim/Output/human/0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)

def main():
    print("Running on simulator data.")
    frames = os.listdir(sim_dir)
    frame_ids = [x.split("_")[1] for x in frames if "_" in x]
    robot_mm = mental_model.MentalModel()
    for frame_id in frame_ids:
        print("Pulling frame", frame_id)
        rgb = cv2.imread(sim_dir + "/Action_" + frame_id + "_0_normal.png")
        depth = cv2.imread(sim_dir + "/Action_" + frame_id + "_0_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]  # exr comes in at HxWx3, we want HxW
        robot_mm.update_from_rgbd_and_pose(rgb, depth, None)
        # cv2.imshow("depth", depth)
        # cv2.waitKey()

    return

if __name__ == "__main__":
    print("Testing the dynamic scene graph on simulator data.")
    main()
    print("Done.")
