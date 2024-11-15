# dsg_sim.py: construct a DSG from a simulator

import mental_model
from pose_estimation import pose
import os, glob, ast, pickle, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.font_manager as fm
import matplotlib.patches
import preprocess_sim_detection
import utils
import visualization.plot_full_tmm
import visualization.plot_pred_human
import prediction.predict

# MAC USERS: Oct 1 2024: MPS is not supported well by PyTorch so we need to enable the fallback, add the following export to your terminal (easiest in ./bashrc)
# export PYTORCH_ENABLE_MPS_FALLBACK=1
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)
plt.rcParams['font.family'] = 'Roboto'  # you probably need to install Roboto -- download the ttf from fonts.google.com. On Mac, place the Roboto folder in Library/Fonts and delete ~/.matplotlib/fontList-vXXX.json. On Linux, run sudo fc-cache -fv
matplotlib.use('QtAgg')

classes = sorted(["human", 'perfume', 'candle', 'bananas', 'cutleryfork', 'washingsponge', 'apple', 'cereal', 'lime', 'cellphone', 'bellpepper', 'crackers', 'garbagecan', 'chips', 'peach', 'toothbrush', 'pie', 'cupcake', 'creamybuns', 'plum', 'chocolatesyrup', 'towel', 'folder', 'toothpaste', 'computer', 'book', 'fryingpan', 'paper', 'mug', 'dishbowl', 'remotecontrol', 'dishwashingliquid', 'cutleryknife', 'plate', 'hairproduct', 'candybar', 'slippers', 'painkillers', 'whippedcream', 'waterglass', 'salmon', 'barsoap', 'character', 'wineglass'])
class_to_class_id = {o : i for i, o in enumerate(classes)}
class_id_to_color_map = matplotlib.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=len(classes)), cmap=matplotlib.cm.hsv).to_rgba([i for i, x in enumerate(classes)])  # color mapper

def main(agent_id="0", episode_dir=None, use_gt_pose=False, use_gt_semantics=False, save_plot=False, show_plot=None):
    # validate episode_dir
    assert episode_dir is not None, "Missing episode_dir param"
    episode_name = episode_dir.split("/")[-1]

    print(f"Running on simulator data, episode: {episode_name}")
    depth_classes = ["human", "person", "human standing", "person standing", "silhouette of a person", "silhouette of a human", "silhouette of a person from the side", "silhouette of a human from the side", "silhouette of a person"]  # not used for ground truth sim data

    # get the ground truth color map, used for ground truth semantics and to initialize the scene
    with open(f"{episode_dir}/episode_info.txt") as f:
        gt_semantic_colormap = ast.literal_eval(f.readlines()[3])

    # get the agent poses
    print("Reading agent poses", f"{episode_dir}/{agent_id}/pd_{episode_name}.txt")
    agent_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, "0")
    human_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, "1")
    print("Done reading poses", len(agent_poses))

    robot_mm = mental_model.MentalModel()  # initialize the robot's mental model
    gt_human_mm = mental_model.MentalModel()  # initialize the ground truth human's mental model
    pred_human_mm = mental_model.MentalModel()  # initialize the predicted human's mental model

    # initialize the environment map, note that this is the "we know the starting layout" assumption
    initial_objects = [{"class": gt_semantic_colormap[k][1], "x": gt_semantic_colormap[k][2][0], "y": gt_semantic_colormap[k][2][2], "z": gt_semantic_colormap[k][2][1]} for k in gt_semantic_colormap if gt_semantic_colormap[k][1] in classes]  # get the original objects

    robot_mm.initialize(objects=initial_objects, verbose=False)  # set the initial environment state
    gt_human_mm.initialize(objects=initial_objects, verbose=False)
    pred_human_mm.initialize(objects=initial_objects, verbose=False)

    # visualization
    if show_plot is not None:
        vis = show_plot(classes, class_to_class_id, class_id_to_color_map, use_gt_semantics)

    # run through frames and update mental models
    frames = sorted([int(x) for x in agent_poses.keys()])
    for frame_id in frames:
        print(f"Processing frame {frame_id}")
        agent_pose = agent_poses[str(frame_id)]
        robot_frame_prefix = f"{episode_dir}/{agent_id}/Action_{str(frame_id).zfill(4)}_0"
        robot_preprocessed_file_path = f"{robot_frame_prefix}_detected.pkl"
        human_frame_prefix = f"{episode_dir}/1/Action_{str(frame_id).zfill(4)}_0"
        human_preprocessed_file_path = f"{human_frame_prefix}_detected.pkl"

        bgr = cv2.imread(f"{robot_frame_prefix}_normal.png")
        robot_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
        depth = cv2.imread(f"{robot_frame_prefix}_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # exr comes in at HxWx3, we want HxW
        depth_1channel = depth[:,:,0]

        # update the robot's mental model
        if use_gt_semantics and os.path.exists(robot_preprocessed_file_path):
            with open(robot_preprocessed_file_path, "rb") as f:
                (agent_pose, robot_detected_objects, _, robot_human_detections) = pickle.load(f)
            robot_human_detections = (robot_human_detections, None, None)  # rgb, depth, filtered
            robot_detected_objects = [x for x in robot_detected_objects if x["class"] in classes]
            robot_mm.update_from_detected_objects(robot_detected_objects)
        else:  # otherwise process the frame now
            gt_semantic = cv2.imread(f"{robot_frame_prefix}_seg_inst.png") if use_gt_semantics else None # if gt_semantic is passed in to the mental model, it will be used. If not, the RGB image will be segmented.
            gt_semantic = cv2.cvtColor(gt_semantic, cv2.COLOR_BGR2RGB) if use_gt_semantics else None
            robot_detected_objects, robot_human_detections = robot_mm.update_from_rgbd_and_pose(robot_rgb, depth_1channel, agent_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, gt_semantic=gt_semantic, gt_semantic_colormap=gt_semantic_colormap, seg_threshold=0.4, seg_save_name="box_bgr_" + str(frame_id).zfill(4))

        # update the ground truth human mental model
        if os.path.exists(human_preprocessed_file_path):
            with open(human_preprocessed_file_path, "rb") as f:
                (gt_human_pose, human_detected_objects, _, human_human_detections) = pickle.load(f)
            human_human_detections = (human_human_detections, None, None)  # rgb, depth, filtered
            human_detected_objects = [x for x in human_detected_objects if x["class"] in classes]
            gt_human_mm.update_from_detected_objects(human_detected_objects)

        objects_visible_to_human = []  # objects that the robot thinks the human can see
        # update the human pose
        if len(robot_human_detections[0]) > 0:  # if a human was seen
            human_pose = human_poses[str(frame_id)] if use_gt_pose else robot_human_detections[0][0]["pose"]  # get the human's pose'
            human_location = [human_pose[0][0], human_pose[0][1], human_pose[0][2]]  # pose[0] is the base joint, using [east, north, vertical]
            human_direction = pose.get_direction_from_pose(human_pose, use_gt_pose=use_gt_pose)  # get the direction that the human is facing
            objects_visible_to_human = robot_mm.get_objects_in_visible_region(human_location, human_direction)  # get the objects that should be in the human's view'
            pred_human_mm.update_from_detected_objects(objects_visible_to_human)  # update the predicted human's mental model
            robot_human_detections[0][0]["pose"] = human_pose  # update the human's pose in the detections
            robot_human_detections[0][0]["direction"] = human_direction  # update the human's direction in the detections
            robot_human_detections[0][0]["visible objects"] = objects_visible_to_human  # update the human's visible objects in the detections

        if save_plot or show_plot is not None:
            if show_plot == visualization.plot_pred_human.PlotPredHuman:
                bgr = cv2.imread(f"{human_frame_prefix}_normal.png")
                human_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
                vis.update(robot_mm, pred_human_mm, gt_human_mm, agent_pose, gt_human_pose, robot_detected_objects, robot_human_detections, objects_visible_to_human, robot_rgb, human_rgb, frame_id)
            elif show_plot == visualization.plot_full_tmm.PlotFullTMM:
                vis.update(robot_mm, pred_human_mm, gt_human_mm, agent_pose, detected_objects, human_detections, rgb, depth, frame_id)
            if save_plot:
                plt.savefig(f"visualization_frames/frame_{frame_id}.png", dpi=300)
            plt.pause(.01)
    return

if __name__ == "__main__":
    map_boundaries = utils.get_map_boundaries("./map_boundaries.png")
    path, all_neighbors = prediction.predict.predict_path([75, 40], [25, 75], map_boundaries)
    prediction.predict.debug_path(map_boundaries, path, all_neighbors)
    print("Testing the dynamic scene graph on simulator data.")
    episode_name = "episode_2024-09-04-16-32_agents_2_run_19"
    episode_dir = f"episodes/{episode_name}"
    # get the agent ID
    agent_id = "0"
    if len(sys.argv) > 1:
        agent_id = sys.argv[1]
    main(agent_id=agent_id, episode_dir=episode_dir, use_gt_pose=True, use_gt_semantics=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman)
    print("Done.")
