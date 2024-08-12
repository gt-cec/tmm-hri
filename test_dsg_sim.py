# dsg_sim.py: construct a DSG from a simulator

import mental_model
import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

sim_dir = "../Output/human/0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)

colormap = matplotlib.cm.hsv

def main(visualize=False):
    print("Running on simulator data.")
    
    # choose classes
    classes = ["person standing", "cup", "oven", "sink", "bottle", "fork", "knife", "fruit", "vegetable", "bottle", "bed", "pillow", "lamp", "book", "trash can", "refrigerator", "bowl", "plant", "television"]
    norm = plt.Normalize(vmin=1, vmax=len(classes))
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    class_to_color_map = scalar_mappable.to_rgba([i for i, x in enumerate(classes)])  # color mapper

    # get the poses
    print("Reading poses")
    poses = {}
    with open(sim_dir + "/pd_human.txt") as f:
        for line in f.readlines()[1:]:
            vals = line.split(" ")
            frame = vals[0]
            hip_loc = extract_pose_loc_for_index(vals, 0, cast_to_numpy_array=True)
            left_shoulder_loc = extract_pose_loc_for_index(vals, 11, cast_to_numpy_array=True)
            right_shoulder_loc = extract_pose_loc_for_index(vals, 12, cast_to_numpy_array=True)
            forward = np.cross(left_shoulder_loc - hip_loc, right_shoulder_loc - hip_loc)  # forward vector
            forward /= np.linalg.norm(forward)
            poses[frame] = [hip_loc, left_shoulder_loc, right_shoulder_loc, forward]

    print("Done reading poses")

    # init the visualization
    if visualize:
        fig = plt.figure()
        # scatter plot axis
        ax_scatter = fig.add_subplot(221, projection='3d')
        ax_scatter.set_xlim((-1000, 1000))
        ax_scatter.set_ylim((-1000, 1000))
        ax_scatter.set_zlim((-1000, 1000))
        xs, ys, zs = [], [], []
        plot_scatter = ax_scatter.scatter(xs, ys, zs)
        # rgb view axis
        ax_rgb = fig.add_subplot(222)
        ax_rgb.set_xlim(0, 512)
        ax_rgb.set_ylim(0, 512)
        ax_rgb.set_aspect('equal')
        ax_rgb.set_axis_off()
        plot_rgb = ax_rgb.imshow(np.zeros((512, 512, 3)))
        # depth view axis
        ax_depth = fig.add_subplot(223)
        ax_depth.set_xlim(0, 512)
        ax_depth.set_ylim(0, 512)
        ax_depth.set_aspect('equal')
        ax_depth.set_axis_off()
        plot_depth = ax_depth.imshow(np.zeros((512, 512, 1)))
        # segmentation view axis
        ax_seg = fig.add_subplot(224)
        ax_seg.set_xlim(0, 512)
        ax_seg.set_ylim(0, 512)
        ax_seg.set_aspect('equal')
        ax_seg.set_axis_off()
        plot_seg = ax_seg.imshow(np.zeros((512, 512, 1)))

    pose_frame_ids = sorted(poses.keys())

    frames = sorted([int(x.split("_")[1]) for x in os.listdir(sim_dir) if x.startswith("Action") and x.endswith(".png")])  # get frames, the .png filter prevents duplicates (each frame has .png and .exr)
    robot_mm = mental_model.MentalModel()
    for frame_id in frames:
        if str(frame_id) not in pose_frame_ids:  # shouldn't happen (means pose data and frame data are misaligned), but just in case
            continue
        print("Pulling frame", frame_id)
        bgr = cv2.imread(sim_dir + "/Action_" + str(frame_id).zfill(4) + "_0_normal.png")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
        depth = cv2.imread(sim_dir + "/Action_" + str(frame_id).zfill(4) + "_0_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # exr comes in at HxWx3, we want HxW
        depth_1channel = depth[:,:,0]
        pose = poses[str(int(frame_id))]
        objects = robot_mm.update_from_rgbd_and_pose(rgb, depth_1channel, pose, classes, 0.4, seg_save_name="box_bgr_" + str(frame_id).zfill(4))

        if visualize:
            x = [o["x"] for o in objects]  # x coordinates for plotting
            y = [o["y"] for o in objects]  # y coordinates for plotting
            z = [o["z"] for o in objects]  # z coordinates for plotting
            seg = np.zeros((rgb.shape[0], rgb.shape[1], 4))  # segmentation colors
            for o in objects:  # get the segmentation by color
                color = class_to_color_map[o["class id"]]
                seg[np.array(o["seg mask"]) > 0] = color

                # add the robot agent
                x.append(pose[0][0])  # right
                y.append(pose[0][2])  # up
                z.append(pose[0][1])  # forward
                agent_c = np.array([[0, 0, 0, 1]])
                if x != [] and y != [] and z != []:
                    plot_scatter._offsets3d = (x, y, z)
                    plot_scatter.set_color(color)

            # update the rgb image
            plot_rgb.set_data(rgb[::-1,:,:])
            # update the depth image
            plot_depth.set_data(depth[::-1,:,:] / 5)
            # update the seg image
            plot_seg.set_data(seg[::-1,:,:])
            plt.pause(.1)

    return

# get the x/y/z coordinates of an index from the pose array
def extract_pose_loc_for_index(pose_list, index, cast_to_numpy_array=False):
    r = [float(pose_list[3 * index + 0]), float(pose_list[3 * index + 1]), float(pose_list[3 * index + 2])]
    return np.array(r) if cast_to_numpy_array else r

if __name__ == "__main__":
    print("Testing the dynamic scene graph on simulator data.")
    main(visualize=True)
    print("Done.")
