# utils.py: functions used by multiple scripts

import numpy as np
import ast, math, os
import cv2

def compute_mean_depth(seg_pixel_locations, depth_map):
    """
    Computes the mean depth of the target class in an image.
    
    Parameters:
        seg_pixel_locations
        depth_map
    
    Returns:
        float: Mean depth value of the target class or None if no pixels are found.
    """

    # Extract depth values at the identified pixel locations
    depth_values = depth_map[seg_pixel_locations]
    
    # Compute the mean depth
    if len(depth_values) > 0:
        return float(np.mean(depth_values))
    else:
        return None

# get the x/y/z coordinates of an index from the pose array
def extract_pose_loc_for_index(pose_list, index, cast_to_numpy_array=False):
    r = [float(pose_list[3 * index + 0]), float(pose_list[3 * index + 1]), float(pose_list[3 * index + 2])]
    return np.array(r)[[0,2,1]] if cast_to_numpy_array else [r[0], r[2], r[1]]  # convert from coordinate frame (east, vertical, north) to (east, north, vertical)

# project visually observed objects from the agent's frame, sim fov = 1.0472rad = 120deg
def project_detected_objects_positions_given_seg_masks_and_agent_pose(detected_objects, agent_pose, seg_masks, depth, fov):
    """
    agent_pose: (robot location, robot heading)
    """
    fov = np.deg2rad(fov)
    # get the location of the object in 3D space
    for i in range(seg_masks.shape[0]):  # for each observed object
        mask = depth[seg_masks[i] > 0]  # the depth field corresponding to the object mask
        dist = mask.sum() / seg_masks[i].sum()  # mean depth
        indices = seg_masks[i].nonzero()  # get indices of the mask
        avg_row = sum(indices[0]) / len(indices[0])  # get average row
        avg_col = sum(indices[1]) / len(indices[1])  # get average col
        horz_angle = (avg_col - seg_masks[i].shape[1] / 2) / seg_masks[i].shape[1] * fov  # get angle left/right from center
        vert_angle = (avg_row - seg_masks[i].shape[0] / 2) / seg_masks[i].shape[0] * fov  # get angle up/down from center
        x_pos_local = math.sin(horz_angle) * dist
        z_pos_local = math.sin(vert_angle) * dist
        direction = agent_pose[1]  # forward vector
        pose_horz_angle = math.atan2(direction[1], direction[0])  # for the pose, z is horz plan up (y), x is horz plan right (x)
        pose_vert_angle = math.asin(direction[2])  # for the pose, y is the vert plan up, so angle is y / 1
        x_pos_global = agent_pose[0][0] + math.cos(pose_horz_angle - horz_angle) * dist
        y_pos_global = agent_pose[0][1] + math.sin(pose_horz_angle - horz_angle) * dist
        z_pos_global = agent_pose[0][2] + math.sin(pose_vert_angle - vert_angle) * dist  # in the future, should use head instead for eye vert
        detected_objects[i]["x"] = float(x_pos_global)
        detected_objects[i]["y"] = float(y_pos_global)
        detected_objects[i]["z"] = float(z_pos_global)
        detected_objects[i]["seg mask"] = seg_masks[i]
        detected_objects[i]["debug"] = {"dist": dist, "pose horz angle": pose_horz_angle, "horz angle": horz_angle, "horz angle global": pose_horz_angle + horz_angle, "x local": x_pos_local, "x global": x_pos_global, "y global": y_pos_global}
    return detected_objects

# get the agent pose in each frame
def get_agent_pose_per_frame(episode_dir:str, episode_name:str, agent:str) -> dict:
    poses = {}
    with open(f"{episode_dir}/{agent}/pd_{episode_name}.txt") as f:
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

# get frame IDs for an agent
def get_frames_filenames(episode_dir:str, episode_name:str, agent:str) -> dict:
    return sorted([int(x.split("_")[1]) for x in os.listdir(f"{episode_dir}/{agent}") if x.startswith("Action") and x.endswith(".png")])  # get frames, the .png filter prevents duplicates (each frame has .png and .exr)

# get the ground truth color map
def get_ground_truth_semantic_colormap(episode_dir):
    with open(f"{episode_dir}/episode_info.txt") as f:
        return ast.literal_eval(f.readlines()[3])

# euclidean distance
def dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + ((a[2] - b[2]) ** 2 if len(a) > 2 else 0)

# read a map image and get boundaries
def get_map_boundaries(map_image_path:str) -> dict:
    map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)  # image must be black/white, with white being the free space
    walls = map_image < 127
    map_image[map_image >= 127] = 0
    map_image[walls] = 1
    map_image = np.transpose(map_image)
    map_image = np.flip(map_image, axis=1)
    return map_image

# calculates the real world coordinates of a point
def calculate_rw_coordinates(root_2d, root_depth, fov, img_res):
    """
    Calculate real-world (RW) coordinates of a joint using simple projection.

    Args:
        root_2d (tuple): The 2D coordinates of the joint (u, v) in pixels.
        root_depth (float): Depth of the root joint in meters (Z).
        fov (float): Field of view (horizontal) in degrees.
        img_res (tuple): Resolution of the image as (width, height).

    Returns:
        tuple: Real-world coordinates (X, Y, Z).
    """
    # Extract image resolution
    img_width, img_height = img_res
    
    # Principal point at the center of the image
    cx, cy = img_width / 2, img_height / 2

    # Compute focal length (in pixels)
    f_x = img_width / (2 * math.tan(math.radians(fov) / 2))
    f_y = f_x  # Assuming square pixels

    # Extract 2D root position
    u, v = root_2d[0], root_2d[1]

    # Calculate real-world coordinates using pinhole projection
    X = (u - cx) * root_depth / f_x
    Y = (v - cy) * root_depth / f_y
    Z = root_depth  # Depth remains unchanged

    return X, Y, Z
