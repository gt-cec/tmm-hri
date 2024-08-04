# dsg_sim.py: construct a DSG from a simulator

import mental_model
import os, glob
import cv2

sim_dir = "../Output/human/0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)

def main():
    print("Running on simulator data.")

    # get the poses
    print("Reading poses")
    poses = {}
    with open(sim_dir + "/pd_human.txt") as f:
        for line in f.readlines()[1:]:
            vals = line.split(" ")
            frame = vals[0]
            print("frame", frame)
            # hip_x = float(vals[1])  # right
            # hip_y = float(vals[2])  # up
            # hip_z = float(vals[3])  # front
            poses[frame] = [float(x) for x in vals[1:]] #[hip_x, hip_y, hip_z]
    print("Done reading poses")

    pose_frame_ids = sorted(poses.keys())

    frames = [x.split("_")[1] for x in os.listdir(sim_dir) if x.startswith("Action")]
    robot_mm = mental_model.MentalModel()
    for frame_id in frames:
        if str(int(frame_id)) not in pose_frame_ids:  # shouldn't happen (means pose data and frame data are misaligned), but just in case
            continue
        print("Pulling frame", frame_id)
        rgb = cv2.imread(sim_dir + "/Action_" + frame_id + "_0_normal.png")
        depth = cv2.imread(sim_dir + "/Action_" + frame_id + "_0_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]  # exr comes in at HxWx3, we want HxW
        robot_mm.update_from_rgbd_and_pose(rgb, depth, poses[str(int(frame_id))])
        # cv2.imshow("depth", depth)
        # cv2.waitKey()

    return

if __name__ == "__main__":
    print("Testing the dynamic scene graph on simulator data.")
    main()
    print("Done.")
