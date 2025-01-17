import cv2
import detection
import os
import utils

def main(episode_name, episode_dir="episodes", agent_id:str="0"):
    seg_image = None
    seg_colormap = utils.get_ground_truth_semantic_colormap(episode_dir + "/" + episode_name)
    agent_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, agent_id)
    for frame in agent_poses.keys():
        file_prefix = f"{episode_dir}/{agent_id}/Action_{frame.zfill(4)}_0"
        seg_image = cv2.imread(f"{file_prefix}_seg_inst.png")
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
        detected_objects, seg_masks = detection.detect.detect_from_ground_truth(seg_image, seg_colormap, classes=[], class_to_class_id=[])
        detected_objects = utils.project_detected_objects_positions_given_seg_masks_and_agent_pose(detected_objects, agent_poses[frame], seg_masks, depth_image, fov=40)  # set the detected objects' locations
            

if __name__ == "__main__":
    main("0", "episode_42")

