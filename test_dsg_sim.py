# dsg_sim.py: construct a DSG from a simulator

import mental_model
import cec_pose
import os, glob, ast, pickle, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.font_manager as fm
import matplotlib.patches
import utils
import visualization.plot_full_tmm
import visualization.plot_pred_human
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


def experiment_parents_are_out(episode_dir:str, agent_id="0", use_gt_human_pose=False, use_gt_semantics=False, use_gt_human_trajectory=False, infer_human_trajectory=True, save_plot=False, show_plot=None, save_dsgs=None):
    """
    Experiment where objects get rearranged while the human is away. The human's mental model (and pred model) is
    initialized to the pre-shuffle state, and the robot's mental model is initialized to the post-shuffle state.

    Args:
        episode_dir (str): The directory of the episode to run the experiment on.
        agent_id (str): The ID of the agent to use, either "0" or "1".
        use_gt_human_pose (bool): Whether to use the ground truth human pose or the detected human pose.
        use_gt_semantics (bool): Whether to use the ground truth semantics or the detected semantics.
        save_plot (bool): Whether to save the plot.
        show_plot: The plot to show (e.g., visualization.plot_pred_human.PlotPredHuman).
        save_dsgs (bool): Whether to save the DSGs.

    Returns:
        None
    """
    # validate episode_dir
    episode_name = episode_dir.split("/")[-1]

    print(f"[\"Parents Are Out\" Scenario] Running on episode: {episode_name}")

    # get the ground truth color map, used for ground truth semantics and to initialize the scene
    color_to_class, initial_shuffled_objects = utils.load_colormap(episode_name)

    classes = ["human", 'perfume', 'candle', 'bananas', 'cutleryfork', 'washingsponge', 'apple', 'cereal',
                'lime', 'cellphone', 'bellpepper', 'crackers', 'garbagecan', 'chips', 'peach', 'toothbrush',
                'pie', 'cupcake', 'creamybuns', 'plum', 'chocolatesyrup', 'towel', 'folder', 'toothpaste',
                'computer', 'book', 'fryingpan', 'paper', 'mug', 'dishbowl', 'remotecontrol', 'dishwashingliquid',
                'cutleryknife', 'plate', 'hairproduct', 'candybar', 'slippers', 'painkillers', 'whippedcream',
                'waterglass', 'salmon', 'barsoap', 'character', 'wineglass']
    classes = sorted(list(set([color_to_class[k] for k in color_to_class if color_to_class[k] in classes])) + ["human"])  # get the classes that are in the colormap
    class_to_class_id = {o : i for i, o in enumerate(classes)}
    class_id_to_color_map = matplotlib.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=len(classes)), cmap=matplotlib.cm.hsv).to_rgba([i for i, x in enumerate(classes)])  # color mapper

    # filter out initial objects that are not in the classes
    initial_shuffled_objects = [x for x in initial_shuffled_objects if x["class"] in classes]

    # load the pre-shuffled objects
    with open(f"{episode_dir}/preshuffle_nodes.pkl", "rb") as f:
        initial_preshuffled_graph = pickle.load(f)
        initial_preshuffled_objects = []
        for node in initial_preshuffled_graph:
            if node["class_name"] not in classes:
                continue
            initial_preshuffled_objects.append({
                "class": node["class_name"],
                "id": node["id"],
                "x": node["bounding_box"]["center"][0],  # east
                "y": node["bounding_box"]["center"][2],  # north
                "z": node["bounding_box"]["center"][1]   # vertical
            })
        
        # just to make sure, add the character to the pre-shuffled objects
        for node in initial_shuffled_objects:
            if node["class"] == "character" and node not in initial_preshuffled_objects:
                initial_preshuffled_objects.append(node)

    # initialize the models
    pose_detector = cec_pose.PoseDetector()  # share this across mental models, it has no state so no data leakage

    # initialize the mental models
    robot_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the robot's mental model
    gt_human_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the ground truth human's mental model
    pred_human_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the predicted human's mental model

    robot_mm.initialize(objects=initial_shuffled_objects, verbose=False)  # set the initial environment state
    gt_human_mm.initialize(objects=initial_preshuffled_objects, verbose=False)
    pred_human_mm.initialize(objects=initial_preshuffled_objects, verbose=False)

    # run through the simulation
    __run_through_simulation__(agent_id,
                            robot_mm,
                            gt_human_mm,
                            pred_human_mm,
                            episode_dir,
                            classes,
                            class_to_class_id,
                            class_id_to_color_map,
                            color_to_class,
                            map_boundaries,
                            use_gt_human_pose=use_gt_human_pose,
                            use_gt_semantics=use_gt_semantics,
                            use_gt_human_trajectory=use_gt_human_trajectory,
                            infer_human_trajectory=infer_human_trajectory,
                            save_plot=save_plot,
                            show_plot=show_plot,
                            save_dsgs=save_dsgs
                        )

def experiment_static_walkabout(episode_dir:str, agent_id="0", use_gt_human_pose=False, use_gt_semantics=False, save_plot=False, show_plot=None, save_dsgs=None):
    # validate episode_dir
    episode_name = episode_dir.split("/")[-1]

    print(f"Running on simulator data, episode: {episode_name}")

    # get the ground truth color map, used for ground truth semantics and to initialize the scene
    color_to_class, initial_objects = utils.load_colormap(episode_name)

    with open(f"{episode_dir}/episode_info.txt") as f:
        classes = ["human", 'perfume', 'candle', 'bananas', 'cutleryfork', 'washingsponge', 'apple', 'cereal', 'lime', 'cellphone', 'bellpepper', 'crackers', 'garbagecan', 'chips', 'peach', 'toothbrush', 'pie', 'cupcake', 'creamybuns', 'plum', 'chocolatesyrup', 'towel', 'folder', 'toothpaste', 'computer', 'book', 'fryingpan', 'paper', 'mug', 'dishbowl', 'remotecontrol', 'dishwashingliquid', 'cutleryknife', 'plate', 'hairproduct', 'candybar', 'slippers', 'painkillers', 'whippedcream', 'waterglass', 'salmon', 'barsoap', 'character', 'wineglass']
        classes = sorted(list(set([color_to_class[k] for k in color_to_class if color_to_class[k] in classes])) + ["human"])  # get the classes that are in the colormap
        class_to_class_id = {o : i for i, o in enumerate(classes)}
        class_id_to_color_map = matplotlib.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=len(classes)), cmap=matplotlib.cm.hsv).to_rgba([i for i, x in enumerate(classes)])  # color mapper

    # filter out initial objects that are not in the classes
    initial_objects = [x for x in initial_objects if x["class"] in classes]

    # initialize the models
    pose_detector = cec_pose.PoseDetector()  # share this across mental models, it has no state so no data leakage

    # initialize the mental models
    robot_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the robot's mental model
    gt_human_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the ground truth human's mental model
    pred_human_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the predicted human's mental model

    robot_mm.initialize(objects=initial_objects, verbose=False)  # set the initial environment state
    gt_human_mm.initialize(objects=initial_objects, verbose=False)
    pred_human_mm.initialize(objects=initial_objects, verbose=False)

    # run through the simulation
    __run_through_simulation__(agent_id,
                            robot_mm,
                            gt_human_mm,
                            pred_human_mm,
                            episode_dir,
                            classes,
                            class_to_class_id,
                            class_id_to_color_map,
                            color_to_class,
                            map_boundaries,
                            use_gt_human_pose=use_gt_human_pose,
                            use_gt_semantics=use_gt_semantics,
                            save_plot=save_plot,
                            show_plot=show_plot,
                            save_dsgs=save_dsgs
                        )

def __run_through_simulation__(agent_id, robot_mm, gt_human_mm, pred_human_mm, episode_dir, classes, class_to_class_id, class_id_to_color_map, color_to_class, map_boundaries, use_gt_human_pose=False, use_gt_semantics=False, use_gt_human_trajectory=False, infer_human_trajectory=True, start_frame=-1, save_plot=False, show_plot=None, save_dsgs=False):
    # declare the depth classes
    depth_classes = ["human", "person", "human standing", "person standing", "silhouette of a person", "silhouette of a human", "silhouette of a person from the side", "silhouette of a human from the side", "silhouette of a person"]  # not used for ground truth sim data

    # get the agent poses
    print("Reading agent poses", f"{episode_dir}/{agent_id}/pd_{episode_name}.txt")
    agent_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, "0")
    human_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, "1")
    print("Done reading poses", len(agent_poses))
    
    # continue from saved if applicable
    start_frame, _ = utils.get_highest_saved_dsgs(episode_dir)

    if start_frame == -1:
        print("No saved DSGs found, starting from the beginning")
        start_frame = 0
    else:
        print(f"Saved DSGs, continuing from frame {start_frame}")
        with open(f"{episode_dir}/DSGs/DSGs_{start_frame}.pkl", "rb") as f:
            dsgs = pickle.load(f)
            robot_mm.dsg = dsgs["robot"]
            gt_human_mm.dsg = dsgs["gt human"]
            pred_human_mm.dsg = dsgs["pred human"]
        start_frame += 1

    last_saw_human = (None, [])  # (frame ID, location) of where the human was last seen by the robot

    # visualization
    if show_plot is not None:
        vis = show_plot(classes, class_to_class_id, class_id_to_color_map, use_gt_semantics)

    # run through frames and update mental models
    frames = sorted([int(x) for x in agent_poses.keys()])
    for frame_id in frames:
        if frame_id < start_frame:
            continue

        # if the frame is not in the human agent's frame, skip it
        if str(frame_id) not in human_poses:
            continue

        # load the frame files
        print(f"Processing frame {frame_id}")
        agent_pose = [agent_poses[str(frame_id)][0], agent_poses[str(frame_id)][-1]]
        robot_frame_prefix = f"{episode_dir}/{agent_id}/Action_{str(frame_id).zfill(4)}_0"
        robot_preprocessed_file_path = f"{robot_frame_prefix}_detected.pkl"
        human_frame_prefix = f"{episode_dir}/{1-int(agent_id)}/Action_{str(frame_id).zfill(4)}_0"

        bgr = cv2.imread(f"{robot_frame_prefix}_normal.png")
        robot_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
        depth = cv2.imread(f"{robot_frame_prefix}_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # exr comes in at HxWx3, we want HxW
        depth_1channel = depth[:,:,0]

        # update the robot's mental model
        # if using ground truth, get the detected objects directly from the simulator files
        if use_gt_semantics:  # if using ground truth but there is no pre-processed pkl file, process the frame now
            print("  Using ground truth segmentation, no pre-processed pkl was found, defaulting to segmentation network")
            gt_instance_image = cv2.imread(f"{robot_frame_prefix}_seg_inst.png") # if gt_semantic is passed in to the mental model, it will be used. If not, the RGB image will be segmented.
            gt_instance_image = cv2.cvtColor(gt_instance_image, cv2.COLOR_BGR2RGB)
            gt_class_image = cv2.imread(f"{robot_frame_prefix}_seg_class.png") # if gt_semantic is passed in to the mental model, it will be used. If not, the RGB image will be segmented.
            gt_class_image = cv2.cvtColor(gt_class_image, cv2.COLOR_BGR2RGB)
            robot_detected_objects, robot_human_detections = robot_mm.update_from_rgbd_and_pose(robot_rgb, depth_1channel, agent_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, gt_instance_image=gt_instance_image, gt_class_image=gt_class_image, gt_class_colormap=color_to_class, seg_threshold=0.4)
        else:  # detect objects using the object detector and segmentation layer
            print("  Using object detection and segmentation network on RGB input.")
            robot_detected_objects, robot_human_detections = robot_mm.update_from_rgbd_and_pose(robot_rgb, depth_1channel, agent_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, detect_threshold=0.4, seg_save_name=f"{episode_dir}/{agent_id}/Action_{str(frame_id).zfill(4)}")
        
        # update the ground truth human mental model, requires the ground truth from the simulator
        print("  Updating ground truth human mental model")
        gt_human_pose = [human_poses[str(frame_id)][0], human_poses[str(frame_id)][-1]]
        gt_human_instance_image = cv2.imread(f"{human_frame_prefix}_seg_inst.png") # if gt_semantic is passed in to the mental model, it will be used. If not, the RGB image will be segmented.
        gt_human_instance_image = cv2.cvtColor(gt_human_instance_image, cv2.COLOR_BGR2RGB)
        gt_human_class_image = cv2.imread(f"{human_frame_prefix}_seg_class.png") # if gt_semantic is passed in to the mental model, it will be used. If not, the RGB image will be segmented.
        gt_human_class_image = cv2.cvtColor(gt_human_class_image, cv2.COLOR_BGR2RGB)
        gt_human_depth = cv2.imread(f"{human_frame_prefix}_depth.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # exr comes in at HxWx3, we want HxW
        gt_human_depth_1channel = gt_human_depth[:,:,0]
        gt_human_mm.update_from_rgbd_and_pose(None, gt_human_depth_1channel, gt_human_pose, classes, gt_class_image=gt_human_class_image, gt_instance_image=gt_human_instance_image, gt_class_colormap=color_to_class, class_to_class_id=class_to_class_id, depth_classes=depth_classes, detect_threshold=0.4, seg_save_name=f"{episode_dir}/{1-int(agent_id)}/Action_{str(frame_id).zfill(4)}")

        # if the human is visible to the robot, run the trajectory prediction
        objects_visible_to_human = []  # objects that the robot thinks the human can see
        human_trajectory_debug = None
        # update the human pose 
        if len(robot_human_detections[0]) > 0:  # if a human was seen, use the first one (can place this in a loop to support multiple humans, but we only have one human mental model in play)
            print("  Human was observed, so updating the predicted human mental model")
            human_pose = human_poses[str(frame_id)] if use_gt_human_pose else robot_human_detections[0][0]["pose"]  # get the human's pose
            human_location = [human_pose[0], human_pose[1], human_pose[2]] # pose[0] is the base joint, using [east, north, vertical]
            if use_gt_human_pose:
                human_direction = robot_mm.pose_detector.get_direction_from_pose(human_pose, use_gt_human_pose=use_gt_human_pose) # get the direction that the human is facing
                human_location = [human_pose[0][0], human_pose[0][1], human_pose[0][2]]
                robot_human_detections[0][0]["pose"] = human_location # update the human's pose in the detections
                robot_human_detections[0][0]["direction"] = human_direction  # update the human's direction in the detections
            robot_human_detections[0][0]["visible objects"] = objects_visible_to_human  # update the human's visible objects in the detections
            
            # if human has not been seen since before the last frame, predict where the human went since the last view
            if last_saw_human[0] is not None:
                # if using ground truth human trajectory, get the poses
                gt_human_poses = []
                if use_gt_human_trajectory:
                    for frame_id in range(last_saw_human[0], frame_id+1):  # for each frame since the last seen
                        if str(frame_id) in human_poses:
                            _human_location = human_poses[str(frame_id)][0]  # get the pose
                            _human_direction = robot_mm.pose_detector.get_direction_from_pose(human_poses[str(frame_id)], use_gt_human_pose=True)[:2]
                            gt_human_poses.append([_human_location, _human_direction])  # save the pose (x/y/z) and direction (x/y)
                # get objects along the path that the human took
                objects_visible_to_human, human_trajectory_debug = prediction.predict.get_objects_visible_from_last_seen(last_saw_human[1][:2], human_location[:2], map_boundaries, robot_mm.dsg, human_fov=gt_human_mm.fov, end_direction=robot_human_detections[0][0]["direction"][:2], use_gt_human_trajectory=use_gt_human_trajectory, gt_human_poses=gt_human_poses, infer_human_trajectory=infer_human_trajectory, debug_tag=f"{frame_id}")
                pred_human_mm.update_from_detected_objects(objects_visible_to_human)  # update the predicted human's mental model
            last_saw_human = (frame_id, human_location)

        # if saving or showing the plot, generate it
        if save_plot or show_plot is not None:
            print("  Generating the visualization frame.")
            if show_plot == visualization.plot_pred_human.PlotPredHuman:
                bgr = cv2.imread(f"{human_frame_prefix}_normal.png")
                if bgr is not None:
                    human_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
                    gt_human_pose = [human_poses[str(frame_id)][0], human_poses[str(frame_id)][-1]]
                    vis.update(robot_mm, pred_human_mm, gt_human_mm, agent_pose, gt_human_pose, robot_detected_objects, robot_human_detections, human_trajectory_debug, objects_visible_to_human, robot_rgb, depth, human_rgb, frame_id)
            elif show_plot == visualization.plot_full_tmm.PlotFullTMM:
                vis.update(robot_mm, pred_human_mm, gt_human_mm, agent_pose, robot_detected_objects, robot_human_detections, objects_visible_to_human, robot_rgb, depth, frame_id)
            if save_plot:
                plt.savefig(f"visualization_frames/frame_{frame_id}.png", dpi=300)
            plt.pause(.01)  # needed for matplotlib to show the plot and not get tripped up
            print("    Saved the visualization frame.")

        # if saving the dynamic scene graphs, save them
        if save_dsgs:
            with open(f"{episode_dir}/DSGs/DSGs_{frame_id}.pkl", "wb") as f:
                pickle.dump({"robot": robot_mm.dsg, "gt human": gt_human_mm.dsg, "pred human": pred_human_mm.dsg, "human is seen": len(robot_human_detections[0]) > 0}, f)
            print(f"  Saved scene graphs for frame {frame_id}")
        
    return

if __name__ == "__main__":
    print("Testing the dynamic scene graph on simulator data.")
    episode_name = "episode_42"
    episode_dir = f"episodes/{episode_name}_short"
    # get the agent ID
    agent_id = "0"
    if len(sys.argv) > 1:
        agent_id = sys.argv[1]

    # Parents are Out (ablation online all): online detection, online pose, A* human traj, GT human mm
    experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=False, use_gt_semantics=False, use_gt_human_trajectory=False, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)
    
    # Parents are Out (ablation online but GT pose): online detection, GT pose, A* human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=True, use_gt_semantics=False, use_gt_human_trajectory=False, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)
    
    # Parents are Out (ablation online but GT detection): GT detection, online pose, A* human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=False, use_gt_semantics=True, use_gt_human_trajectory=False, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)
    
    # Parents are Out (ablation online but GT path inference): online detection, online pose, GT human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=False, use_gt_semantics=False, use_gt_human_trajectory=True, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)

    # Parents are Out (ablation online but NO path inference): online detection, online pose, NO human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=False, use_gt_semantics=False, use_gt_human_trajectory=False, infer_human_trajectory=False, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)


    # Parents are Out (ablation full GT): GT pose, GT detection, GT human traj, GT traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=True, use_gt_semantics=True, use_gt_human_trajectory=True, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)

    # Parents are Out (ablation full GT but online pose): online pose, GT detection, GT human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=False, use_gt_semantics=True, use_gt_human_trajectory=True, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)

    # Parents are Out (ablation full GT but online detection): GT pose, online detection, GT human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=True, use_gt_semantics=False, use_gt_human_trajectory=True, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)
    
    # Parents are Out (ablation full GT but A* path inference): GT pose, GT detection, online human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=True, use_gt_semantics=True, use_gt_human_trajectory=False, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)

    # Parents are Out (ablation full GT but NO path inference): GT pose, GT detection, online human traj, GT human mm
    # experiment_parents_are_out(episode_dir, agent_id=agent_id, use_gt_human_pose=True, use_gt_semantics=True, use_gt_human_trajectory=False, infer_human_trajectory=True, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)

    # Static Walkabout: Online Robot, GT Human
    #experiment_static_walkabout(episode_dir, agent_id=agent_id, use_gt_human_pose=False, use_gt_semantics=False, save_plot=True, show_plot=visualization.plot_pred_human.PlotPredHuman, save_dsgs=True)
    print("Done.")
