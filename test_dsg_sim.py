# dsg_sim.py: construct a DSG from a simulator

import mental_model
import os, glob, ast
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.font_manager as fm
import matplotlib.patches

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)

colormap = matplotlib.cm.hsv  # for the semantic segmentation
plt.rcParams['font.family'] = 'Roboto'

def main(episode_dir=None, visualize=False):
    # validate episode_dir
    assert episode_dir is not None, "Missing episode_dir param"
    episode_name = episode_dir.split("/")[-1]

    print(f"Running on simulator data, episode: {episode_name}")
    
    # choose classes
    classes = ["human", "cup", "oven", "sink", "bottle", "fork", "knife", "fruit", "vegetable", "bottle", "bed", "pillow", "lamp", "book", "trash can", "refrigerator", "plate", "bowl", "plant", "tvstand"]
    class_to_class_id = {o : i for i, o in enumerate(classes)}
    depth_classes = ["human", "person", "human standing", "person standing", "silhouette of a person", "silhouette of a human", "silhouette of a person from the side", "silhouette of a human from the side", "silhouette of a person"]
    norm = plt.Normalize(vmin=1, vmax=len(classes))
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    class_to_color_map = scalar_mappable.to_rgba([i for i, x in enumerate(classes)])  # color mapper

    # get the ground truth color map
    with open(f"{episode_dir}/episode_info.txt") as f:
        gt_semantic_colormap = ast.literal_eval(f.readlines()[3])

    # get the agent poses
    print("Reading agent poses", f"{episode_dir}/0/pd_{episode_name}.txt")
    poses = {}
    with open(f"{episode_dir}/0/pd_{episode_name}.txt") as f:
        for line in f.readlines()[1:]:
            vals = line.split(" ")
            frame = vals[0]
            vals = vals[1:]
            hip_loc = extract_pose_loc_for_index(vals, 0, cast_to_numpy_array=True)
            left_shoulder_loc = extract_pose_loc_for_index(vals, 11, cast_to_numpy_array=True)
            right_shoulder_loc = extract_pose_loc_for_index(vals, 12, cast_to_numpy_array=True)
            forward = np.cross(right_shoulder_loc - hip_loc, left_shoulder_loc - hip_loc)  # forward vector
            forward /= np.linalg.norm(forward)
            poses[frame] = [hip_loc, left_shoulder_loc, right_shoulder_loc, forward]

    print("Done reading poses", len(poses))

    # init the visualization
    if visualize:
        fig = plt.figure()
        fig.tight_layout()
        # scatter plot axis
        ax_scatter = fig.add_subplot(221)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        ax_scatter.set_xlim((-10, 10))
        ax_scatter.set_ylim((-10, 10))
        ax_scatter.set_aspect('equal')
        ax_scatter.set_axis_off()
        ax_scatter.set_title("Bird's Eye Semantic Map")
        xs, ys = [], []
        plot_scatter = ax_scatter.scatter(xs, ys)
        # pose lines
        pose_lines = []
        # rgb view axis
        ax_rgb = fig.add_subplot(222)
        ax_rgb.set_xlim(0, 512)
        ax_rgb.set_ylim(0, 512)
        ax_rgb.set_aspect('equal')
        ax_rgb.set_axis_off()
        ax_rgb.set_title("Robot RGB Camera")
        plot_rgb = ax_rgb.imshow(np.zeros((512, 512, 3)))
        # segmentation view axis
        ax_seg = fig.add_subplot(223)
        ax_seg.set_xlim(0, 512)
        ax_seg.set_ylim(0, 512)
        ax_seg.set_aspect('equal')
        ax_seg.set_axis_off()
        ax_seg.set_title("Camera Segmentation")
        plot_seg = ax_seg.imshow(np.zeros((512, 512, 1)))
        # depth view axis
        ax_depth = fig.add_subplot(224)
        ax_depth.set_xlim(0, 512)
        ax_depth.set_ylim(0, 512)
        ax_depth.set_aspect('equal')
        ax_depth.set_axis_off()
        ax_depth.set_title("Robot Depth Camera")
        plot_depth = ax_depth.imshow(np.zeros((512, 512, 1)))
        # annotate the figure
        fig.text(0.01, 0.99, "Demo of Semantic Map Construction", ha='left', va='top', fontsize=14, color='black')
        classes_vert = 0.90
        fig.text(0.01, classes_vert, "Classes:", ha='left', va='top', fontsize=10, color='black')
        for i, c in enumerate(classes):
            classes_vert -= 0.03
            fig.text(0.01, classes_vert, "    ҉  " + c, ha='left', va='top', fontsize=8, color=class_to_color_map[i])
        text_frame = fig.text(0.99, 0.01, "Frame: N/A", ha="right", va="bottom", fontsize=8, color="black")

    pose_frame_ids = sorted(poses.keys())

    frames = sorted([int(x.split("_")[1]) for x in os.listdir(episode_dir + "/0") if x.startswith("Action") and x.endswith(".png")])  # get frames, the .png filter prevents duplicates (each frame has .png and .exr)
    
    robot_mm = mental_model.MentalModel()  # initialize the robot's mental model

    initial_objects = [{"class": gt_semantic_colormap[k][1], "x": gt_semantic_colormap[k][2][0], "y": gt_semantic_colormap[k][2][1], "z": gt_semantic_colormap[k][2][2]} for k in gt_semantic_colormap if gt_semantic_colormap[k][1] in classes]  # get the original objects
    robot_mm.initialize(objects=initial_objects, verbose=True)  # set the initial environment state

    # run through frames and update mental models
    for frame_id in frames:
        if str(frame_id) not in pose_frame_ids:  # shouldn't happen (means pose data and frame data are misaligned), but just in case
            continue
        bgr = cv2.imread(episode_dir + "/0/Action_" + str(frame_id).zfill(4) + "_0_normal.png")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
        depth = cv2.imread(episode_dir + "/0/Action_" + str(frame_id).zfill(4) + "_0_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # exr comes in at HxWx3, we want HxW
        depth_1channel = depth[:,:,0]
        gt_semantic = cv2.imread(episode_dir + "/0/Action_" + str(frame_id).zfill(4) + "_0_seg_inst.png")  # if gt_semantic is passed in, it will be used. If not, the RGB image will be segmented.
        gt_semantic = cv2.cvtColor(gt_semantic, cv2.COLOR_BGR2RGB)
        pose = poses[str(int(frame_id))]
        objects, human_detections = robot_mm.update_from_rgbd_and_pose(rgb, depth_1channel, pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, gt_semantic=gt_semantic, gt_semantic_colormap=gt_semantic_colormap, seg_threshold=0.4, seg_save_name="box_bgr_" + str(frame_id).zfill(4))

        if visualize:           
            # add the robot agent
            x, y, z = [], [], []
            x.append(pose[0][0])  # right
            y.append(pose[0][2])  # forward
            z.append(pose[0][1])  # up
            plot_colors = [(0, 0, 0, 1)] # initialize with the agent pose

            # fade out the previous motion colors
            for i, line in enumerate(pose_lines):
                alpha = line.get_alpha()
                if alpha is not None:  # if the line has an alpha value
                    if alpha <= 0.05:  # delete if going invisible
                        del pose_lines[i]
                        continue
                    line.set_alpha(alpha - 0.05)  # decrease alpha

            seg = np.zeros((rgb.shape[0], rgb.shape[1], 4))  # segmentation colors
            for o in objects:  # get the segmentation by color
                color = class_to_color_map[o["class id"]]
                seg[np.array(o["seg mask"]) > 0] = color
                x.append(o["x"])
                y.append(o["y"])
                plot_colors.append(color)

            if x != [] and y != [] and z != []:  # update the plot points and colors
                plot_scatter.set_offsets(np.c_[x, y])
                plot_scatter.set_color(plot_colors)
            line = ax_scatter.axes.plot((pose[0][0], pose[0][0] + pose[-1][0]), (pose[0][2], pose[0][2] + pose[-1][2]), color=(0, 0, 0), alpha=1.0)
            pose_lines.append(line[0])

            # update the rgb image
            plot_rgb.set_data(rgb[::-1,:,:])
            # plot_rgb.set_data(gt_semantic[::-1,:,:])
            for patch in ax_rgb.patches + ax_rgb.texts:  # remove existing rectangles
                patch.remove()
            for rgb_human in human_detections[0]:  # add rectangles for detected humans
                ax_rgb.add_patch(matplotlib.patches.Rectangle((rgb_human["box"][0], rgb.shape[1] - rgb_human["box"][1]), rgb_human["box"][2] - rgb_human["box"][0], rgb_human["box"][1] - rgb_human["box"][3], linewidth=1, edgecolor='r', facecolor='none'))
                ax_rgb.text(rgb_human["box"][0], rgb.shape[1] - rgb_human["box"][1], rgb_human["confidence"], fontfamily="sans-serif", fontsize=4, color="white", bbox=dict(facecolor="red", linewidth=1, alpha=1.0, edgecolor="red", pad=0))                
            # update the depth image
            plot_depth.set_data(depth[::-1,:,:] / 5)
            for patch in ax_depth.patches + ax_depth.texts:  # remove existing rectangles
                patch.remove()
            for depth_human in human_detections[1]:  # add rectangles for detected humans
                ax_depth.add_patch(matplotlib.patches.Rectangle((depth_human["box"][0], depth.shape[1] - depth_human["box"][1]), depth_human["box"][2] - depth_human["box"][0], depth_human["box"][1] - depth_human["box"][3], linewidth=1, edgecolor='r', facecolor='none'))
                ax_depth.text(depth_human["box"][0], depth.shape[1] - depth_human["box"][1], depth_human["confidence"], fontfamily="sans-serif", fontsize=4, color="white", bbox=dict(facecolor="red", linewidth=1, alpha=1.0, edgecolor="red", pad=0))                
            # update the seg image
            plot_seg.set_data(seg[::-1,:,:])
            text_frame.set_text(f"Frame: {frame_id}")
            plt.savefig(f"frames/frame_{frame_id}.png", dpi=300)
            plt.pause(.01)

    return

# get the x/y/z coordinates of an index from the pose array
def extract_pose_loc_for_index(pose_list, index, cast_to_numpy_array=False):
    r = [float(pose_list[3 * index + 0]), float(pose_list[3 * index + 1]), float(pose_list[3 * index + 2])]
    return np.array(r) if cast_to_numpy_array else r

if __name__ == "__main__":
    print("Testing the dynamic scene graph on simulator data.")
    episode_name = "episode_2024-09-04-16-32_agents_2_run_19"
    episode_dir = f"episodes/{episode_name}"
    main(episode_dir=episode_dir, visualize=True)
    print("Done.")
