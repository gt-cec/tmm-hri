# TODO: merge this with utils.py in the parent folder

import numpy as np
import cv2
import pyexr
import math

def compute_mean_depth(seg_map_path, depth_map_path, target_value):
    """
    Computes the mean depth of the target class in an image.
    
    Parameters:
        seg_map_path (str): File path to the segmentation map (PNG).
        depth_map_path (str): File path to the depth map (EXR).
        target_value (list): RGB values identifying the target class in the segmentation map.
    
    Returns:
        float: Mean depth value of the target class or None if no pixels are found.
    """
    # Load the segmentation map
    seg_map = cv2.imread(seg_map_path, cv2.IMREAD_UNCHANGED)
    if seg_map is None:
        raise ValueError(f"Failed to load segmentation map from {seg_map_path}")
    
    # Identify pixels corresponding to the target value
    binary_mask = np.all(seg_map == target_value, axis=-1)
    pixel_locations = np.where(binary_mask)

    # Load the depth map
    try:
        exr_data = pyexr.open(depth_map_path)
        depth_array = exr_data.get("R")  # Adjust channel as needed
    except Exception as e:
        raise ValueError(f"Error loading depth map from {depth_map_path}: {e}")

    # Extract depth values at the identified pixel locations
    depth_values = depth_array[pixel_locations]
    
    # Compute the mean depth
    if len(depth_values) > 0:
        return float(np.mean(depth_values))
    else:
        return None

# get the x/y/z coordinates of an index from the pose array
def extract_pose_loc_for_index(pose_list, index, cast_to_numpy_array=False):
    r = [float(pose_list[3 * index + 0]), float(pose_list[3 * index + 1]), float(pose_list[3 * index + 2])]
    return np.array(r)[[0,2,1]] if cast_to_numpy_array else [r[0], r[2], r[1]]  # convert from coordinate frame (east, vertical, north) to (east, north, vertical)

# get the agent pose in each frame
def get_agent_pose_per_frame(filepath) -> dict:
    poses = {}
    with open(filepath) as f:
        lines = f.readlines()[1:]
        for line in lines:
            vals = line.split(" ")
            frame = vals[0]
            vals = vals[1:]
            hip_loc = extract_pose_loc_for_index(vals, 0, cast_to_numpy_array=True)
            left_shoulder_loc = extract_pose_loc_for_index(vals, 11, cast_to_numpy_array=True)
            right_shoulder_loc = extract_pose_loc_for_index(vals, 12, cast_to_numpy_array=True)
            left_hand_loc = extract_pose_loc_for_index(vals, 17, cast_to_numpy_array=True)  # thumb proximal
            right_hand_loc = extract_pose_loc_for_index(vals, 18, cast_to_numpy_array=True)
            left_foot = extract_pose_loc_for_index(vals, 5, cast_to_numpy_array=True)  # thumb proximal
            right_foot = extract_pose_loc_for_index(vals, 6, cast_to_numpy_array=True)
            head = extract_pose_loc_for_index(vals, 10, cast_to_numpy_array=True)
            forward = -1 * np.cross(right_shoulder_loc - hip_loc, left_shoulder_loc - hip_loc)  # forward vector
            forward /= np.linalg.norm(forward)
            poses[frame] = [hip_loc, left_shoulder_loc, right_shoulder_loc, left_hand_loc, right_hand_loc, left_foot, right_foot, head, forward]
    return poses
