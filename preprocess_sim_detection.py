# preprocess_sim_detection.py: runs through simulation frames to get the observed objects and segmentation maps, for use in later model training

import pickle  # for storing observed objects
import detection.detect  # for detecting objects
import utils  # for common functions
import cv2  # for image reading
import os, os.path  # for some functions

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)

# locates objects in all frames of an episode
def process_sim_run(episode_name, episode_dir):
    print("Reading frame segmentations")
    seg_image = None
    depth_image = None
    seg_colormap = utils.get_ground_truth_semantic_colormap(episode_dir)
    for agent in ["0", "1"]:
        feet_locations = [[0,0,0], [0,0,0]]  # locations of the feet so we know when to freeze the pose (when feet aren't moving')
        agent_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, agent)
        previous_upright_pose = None
        frames_currently_not_moved = 0  # keep track of the number of frames that the feet don't move so we can trigger a standing mode'
        for frame in agent_poses.keys():
            file_prefix = f"{episode_dir}/{agent}/Action_{frame.zfill(4)}_0"
            output_filename = f"{file_prefix}_detected.pkl"
            if os.path.exists(output_filename):
                print(f"Frame {file_prefix} has already been processed")
                continue
            print(f"Processing frame {file_prefix}")
            seg_image = cv2.imread(f"{file_prefix}_seg_inst.png")
            seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)  # opencv reads as bgr, convert to rgb
            depth_image = cv2.imread(f"{file_prefix}_depth.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]  # read depth and pull one channel (2D grayscale)
            detected_objects, seg_masks = detection.detect.detect_from_ground_truth(seg_image, seg_colormap, classes=[], class_to_class_id=[])
            # when the agent grabs something, some bones are closer/further than usual, can use this to ignore these forward vectors as the first person camera does not change orientation during these animations
            h2h = round(utils.dist_sq(agent_poses[frame][3], agent_poses[frame][4]), 3) > .35  # threshold for grabbing
            h2f = round(utils.dist_sq(agent_poses[frame][7], agent_poses[frame][6]), 3) < 2.15  # threshold for grabbing
            left_foot_moved = round(utils.dist_sq(agent_poses[frame][5], feet_locations[0]), 3) > 0.01
            right_foot_moved = round(utils.dist_sq(agent_poses[frame][6], feet_locations[1]), 3) > 0.01
            feet_locations[0] = agent_poses[frame][5]
            feet_locations[1] = agent_poses[frame][6]
            print("moving?", left_foot_moved or right_foot_moved)
            if previous_upright_pose is None or int(frame) < 2:
                previous_upright_pose = agent_poses[frame]
            elif not left_foot_moved and not right_foot_moved:
                frames_currently_not_moved += 1
                if frames_currently_not_moved > 10:
                    agent_poses[frame] = previous_upright_pose
                    print("Standing, so setting agent pose to previous", frame)
                else:
                    previous_upright_pose = agent_poses[frame]
                    print("Feet aren't moving, but not 10 frames yet so still changing post")
            else:
                frames_currently_not_moved = 0
                previous_upright_pose = agent_poses[frame]
                print("Walking, so setting previous pose to agent pose", frame)
            detected_objects = utils.project_detected_objects_positions_given_seg_masks_and_agent_pose(detected_objects, agent_poses[frame], seg_masks, depth_image, fov=40)  # set the detected objects' locations
            for i in range(len(detected_objects)):
                detected_objects[i]["seg mask"] = seg_masks[i]
            human_detections = [x for x in detected_objects if x["class"] == "character"]  # get the human detections so we don't have to get them live

            with open(f"{output_filename}", "wb") as f:
                pickle.dump((agent_poses[frame], detected_objects, seg_masks, human_detections), f)


if __name__ == "__main__":
    episode_name = "episode_2024-09-04-16-32_agents_2_run_19"
    episode_dir = f"episodes/{episode_name}"
    process_sim_run(episode_name, episode_dir)
