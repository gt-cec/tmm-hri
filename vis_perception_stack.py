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
import utils
import visualization.plot_perception
import prediction.predict
import sys

np.set_printoptions(threshold=sys.maxsize)

# MAC USERS: Oct 1 2024: MPS is not supported well by PyTorch so we need to enable the fallback, add the following export to your terminal (easiest in ./bashrc)
# export PYTORCH_ENABLE_MPS_FALLBACK=1
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)
plt.rcParams['font.family'] = 'Roboto'  # you probably need to install Roboto -- download the ttf from fonts.google.com. On Mac, place the Roboto folder in Library/Fonts and delete ~/.matplotlib/fontList-vXXX.json. On Linux, run sudo fc-cache -fv
matplotlib.use('Agg')
plt.ioff()

map_boundaries = utils.get_map_boundaries("./map_boundaries.png")
object_classes = ["human", 'perfume', 'candle', 'bananas', 'cutleryfork', 'washingsponge', 'apple', 'cereal', 'lime', 'cellphone', 'bellpepper', 'crackers', 'garbagecan', 'chips', 'peach', 'toothbrush', 'pie', 'cupcake', 'creamybuns', 'plum', 'chocolatesyrup', 'towel', 'folder', 'toothpaste', 'computer', 'book', 'fryingpan', 'paper', 'mug', 'dishbowl', 'remotecontrol', 'dishwashingliquid', 'cutleryknife', 'plate', 'hairproduct', 'candybar', 'slippers', 'painkillers', 'whippedcream', 'waterglass', 'salmon', 'barsoap', 'character', 'wineglass']

save_dir = "visualization_perception"

def main(episode_dir, agent_id="0", use_gt_human_pose=False, use_gt_semantics=False, save_dsgs=None):
    # validate episode_dir
    episode_name = episode_dir.split("/")[-1]

    print(f"Running on simulator data, episode: {episode_name}")
    depth_classes = ["human", "person", "human standing", "person standing", "silhouette of a person", "silhouette of a human", "silhouette of a person from the side", "silhouette of a human from the side", "silhouette of a person"]  # not used for ground truth sim data

    # get the ground truth color map, used for ground truth semantics and to initialize the scene
    with open(f"{episode_dir}/episode_info.txt") as f:
        gt_semantic_colormap = ast.literal_eval(f.readlines()[3])
        classes = sorted(list(set([gt_semantic_colormap[k][1] for k in gt_semantic_colormap if gt_semantic_colormap[k][1] in object_classes])) + ["human"])  # get the classes that are in the colormap
        class_to_class_id = {o : i for i, o in enumerate(classes)}
        class_id_to_color_map = matplotlib.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=len(classes)), cmap=matplotlib.cm.hsv).to_rgba([i for i, x in enumerate(classes)])  # color mapper

    # get the agent poses
    agent_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, agent_id)

    # initialize the environment map, note that this is the "we know the starting layout" assumption
    initial_objects = [{"class": gt_semantic_colormap[k][1], "x": gt_semantic_colormap[k][2][0], "y": gt_semantic_colormap[k][2][2], "z": gt_semantic_colormap[k][2][1]} for k in gt_semantic_colormap if gt_semantic_colormap[k][1] in classes]  # get the original objects

    # initialize the models
    pose_detector = pose.PoseDetection()  # share this across mental models, it has no state so no data leakage

    # initialize the mental models
    robot_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the robot's mental model
    robot_mm.initialize(objects=initial_objects, verbose=False)  # set the initial environment state

    # initialize the plot
    vis = visualization.plot_perception.PlotPerception(classes, class_to_class_id, class_id_to_color_map, use_gt_semantics)

    # run through frames and update mental models
    frames = sorted([int(x) for x in agent_poses.keys()])
    for frame_id in frames:
        # load the frame files
        print(f"Processing frame {frame_id}")
        agent_pose = [agent_poses[str(frame_id)][0], agent_poses[str(frame_id)][-1]]
        robot_frame_prefix = f"{episode_dir}/{agent_id}/Action_{str(frame_id).zfill(4)}_0"
        robot_preprocessed_file_path = f"{robot_frame_prefix}_detected.pkl"

        bgr = cv2.imread(f"{robot_frame_prefix}_normal.png")
        robot_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
        depth = cv2.imread(f"{robot_frame_prefix}_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # exr comes in at HxWx3, we want HxW
        depth_1channel = depth[:,:,0]

        # update the robot's mental model
        # if using ground truth, get the detected objects directly from the simulator files
        if use_gt_semantics and os.path.exists(robot_preprocessed_file_path):
            print("  Using ground truth segmentation and pre-processed pkl:", robot_preprocessed_file_path)
            with open(robot_preprocessed_file_path, "rb") as f:
                (agent_pose, robot_detected_objects, _, robot_human_detections) = pickle.load(f)
                for i in range(len(robot_human_detections)):
                    robot_human_detections[i]["seg mask"] = np.where(robot_human_detections[i]["seg mask"] == 1)
                agent_pose = (agent_pose[0], agent_pose[-1])  # agent pose is (hip location XY, direction)
            robot_detected_objects = [x for x in robot_detected_objects if x["class"] in classes]
            robot_mm.update_from_detected_objects(robot_detected_objects)
        elif use_gt_semantics:  # if using ground truth but there is no pre-processed pkl file, process the frame now
            print("  Using ground truth segmentation, no pre-processed pkl was found, will detect from semantic image")
            gt_semantic = cv2.imread(f"{robot_frame_prefix}_seg_inst.png") # if gt_semantic is passed in to the mental model, it will be used. If not, the RGB image will be segmented.
            gt_semantic = cv2.cvtColor(gt_semantic, cv2.COLOR_BGR2RGB)
            robot_detected_objects, robot_human_detections = robot_mm.update_from_rgbd_and_pose(robot_rgb, depth_1channel, agent_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, gt_semantic=gt_semantic, gt_semantic_colormap=gt_semantic_colormap, seg_threshold=0.4, seg_save_name="box_bgr_" + str(frame_id).zfill(4))
        else:  # detect objects using the object detector and segmentation layer
            print("  Using object detection and segmentation network on RGB input.")
            robot_detected_objects, robot_human_detections = robot_mm.update_from_rgbd_and_pose(robot_rgb, depth_1channel, agent_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, seg_threshold=0.4, seg_save_name="box_bgr_" + str(frame_id).zfill(4))

        # if the human is visible to the robot, run the trajectory prediction
        objects_visible_to_human = []  # objects that the robot thinks the human can see
        # update the human pose 
        if len(robot_human_detections[0]) > 0:  # if a human was seen, use the first one (can place this in a loop to support multiple humans, but we only have one human mental model in play)
            human_pose = robot_human_detections[0][0]["pose"]  # get the human's pose
            human_location = [human_pose[0], human_pose[1], human_pose[2]] # pose[0] is the base joint, using [east, north, vertical]
            if use_gt_human_pose:
                human_direction = pose.get_direction_from_pose(human_pose, use_gt_human_pose=use_gt_human_pose) # get the direction that the human is facing
                robot_human_detections[0][0]["pose"] = human_pose  # update the human's pose in the detections
                robot_human_detections[0][0]["direction"] = human_direction  # update the human's direction in the detections
            robot_human_detections[0][0]["visible objects"] = objects_visible_to_human  # update the human's visible objects in the detections

        # if saving or showing the plot, generate it
        vis.update(robot_mm, agent_pose, robot_detected_objects, robot_human_detections, robot_rgb, depth, frame_id)
        plt.savefig(f"{save_dir}/frame_{frame_id}.png", dpi=300)
        plt.show()
        plt.pause(.01)  # needed for matplotlib to show the plot and not get tripped up
    return

if __name__ == "__main__":
    print("Testing the dynamic scene graph on simulator data.")
    # episode_name = "episode_2024-09-04-16-32_agents_2_run_19"
    episode_name = "episode_42"
    episode_dir = f"episodes/{episode_name}"
    # get the agent ID
    agent_id = "0"
    if len(sys.argv) > 1:
        agent_id = sys.argv[1]
    
    # make the folder if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), f"{save_dir}/")
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    main(episode_dir, agent_id=agent_id, use_gt_human_pose=False, use_gt_semantics=False, save_dsgs=True)
    print("Done.")
