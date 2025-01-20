# utils.py: functions used by multiple scripts

import numpy as np
import ast, math, os
import cv2
import pickle

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
        y_pos_local = math.cos(horz_angle) * dist
        z_pos_local = math.sin(vert_angle) * dist
        direction = agent_pose[1] / np.linalg.norm(agent_pose[1])  # forward vector
        pose_horz_angle = math.atan2(direction[1], direction[0])  # for the pose, z is horz plan up (y), x is horz plan right (x)
        pose_vert_angle = math.asin(direction[2])  # for the pose, y is the vert plan up, so angle is y / 1
        x_pos_global = agent_pose[0][0] + math.cos(pose_horz_angle - horz_angle) * dist
        y_pos_global = agent_pose[0][1] + math.sin(pose_horz_angle - horz_angle) * dist
        z_pos_global = agent_pose[0][2] + math.sin(pose_vert_angle - vert_angle) * dist  # in the future, should use head instead for eye vert
        detected_objects[i]["x"] = float(x_pos_global)
        detected_objects[i]["x local"] = float(x_pos_local)
        detected_objects[i]["y"] = float(y_pos_global)
        detected_objects[i]["y local"] = float(y_pos_local)
        detected_objects[i]["z"] = float(z_pos_global)
        # detected_objects[i]["seg mask"] = seg_masks[i]
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

# squared euclidean distance
def dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + ((a[2] - b[2]) ** 2 if len(a) > 2 else 0)

# euclidean distance
def dist(a, b):
    return math.sqrt(dist_sq(a, b))

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

# get the highest frame of the saved DSGs
def get_highest_saved_dsgs(episode_dir:str) -> int:
    # create the DSGs directory if it doesn't exist
    if not os.path.exists(f"{episode_dir}/DSGs"):
        os.makedirs(f"{episode_dir}/DSGs")
    dsgs = [x for x in os.listdir(f"{episode_dir}/DSGs") if x.startswith("DSGs_")]
    dsg_ints = [int(x.split("_")[1].split(".")[0]) for x in dsgs]
    if len(dsg_ints) == 0:
        return -1, ()
    max_dsg_i = dsg_ints.index(max(dsg_ints))
    return dsg_ints[max_dsg_i], dsgs[max_dsg_i]

# decide whether a node is applicable or not
def __node_is_grabbable__(n:dict, ignore_objects=(), filter_grabbable=True) -> bool:
    """
    Determines whether a node is a grabbable object.
    """
    if n["category"] != "Props":
        return False
    if filter_grabbable and "GRABBABLE" not in n["properties"]:
        return False
    if n["class_name"] in ignore_objects:
        return False
    return True

def get_mode(img):
    """
    Returns the mode pixel in an image, from https://stackoverflow.com/questions/43826089
    """
    unq,count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
    return unq[count.argmax()]

def load_colormap(episode_name:str, episode_dir:str="episodes"):
    """
    Generates and loads the color map.

    Args:
        episode_name (str): Name of the episode.
        episode_dir (str): Directory where the episode is stored.

    Returns:
        dict: Map from color to object class.
    """
    (_, object_colors, g) = pickle.load(open(f"{episode_dir}/{episode_name}/color_info.pkl", "rb"))
    ignore_classes = ("floor", "wall", "curtains", "ceiling", "bedroom", "kitchencabinet", "nightstand", "washingmachine", "towelrack", "bathtub", "stall", "toilet", "faucet", "chair", "bed", "pillow", "sink", "bookshelf", "fridge", "sofa", "kitchencounterdrawer", "door", "kitchentable", "bathroomcounter", "bathroomcabinet", "window", "stove", "ceilinglamp", "desk", "closet", "tablelamp", "coffeetable", "bathroom", "livingroom", "stovefan", "powersocket", "lightswitch", "wallshelf", "kitchen", "bench", "rug", "walllamp", "photoframe", "wallpictureframe", "tvstand")

    initial_objects = []

    # node_ids = {}
    inst_color_to_candidate_classes = {}
    for i in range(len(g["nodes"])):
        node = g["nodes"][i]
        color = [round(float(x) * 255) for x in object_colors[str(node["id"])]]
        if node["class_name"] in ignore_classes:
            continue
        if str(color) not in inst_color_to_candidate_classes:
            inst_color_to_candidate_classes[str(color)] = []
        inst_color_to_candidate_classes[str(color)].append(node["class_name"])
        initial_objects.append({
            "class": node["class_name"],
            "id": node["id"],
            "x": node["bounding_box"]["center"][0],  # east
            "y": node["bounding_box"]["center"][2],  # north
            "z": node["bounding_box"]["center"][1]   # vertical
        })
        # print("adding", "raw", object_colors[str(node["id"])], "affixed", color, "info", inst_color_to_candidate_classes[str(color)])
    # load the episode's class color map if it exists
    if os.path.exists(f"{episode_dir}/{episode_name}/handcrafted_colormap.txt"):
        with open(f"{episode_dir}/{episode_name}/handcrafted_colormap.txt") as f:
            class_color_to_object_class = ast.literal_eval(f.read())
    elif os.path.exists(f"{episode_dir}/{episode_name}/class_colormap.pkl"):
        with open(f"{episode_dir}/{episode_name}/class_colormap.pkl", "rb") as f:
            class_color_to_object_class = pickle.load(f)
    else:
        print("Creating class color map, this is not provided from virtualhome so you will have to manually verify conflicts.")
        class_color_to_object_class = {}
        # load each frame and get the color map
        for frame in sorted([x for x in os.listdir(f"{episode_dir}/{episode_name}/0/") if x.startswith("Action") and x.endswith("seg_inst.png")]):
            # get the instance image and the class image
            print("Parsing frame", frame)
            frame_inst = cv2.imread(f"{episode_dir}/{episode_name}/0/{frame}")
            frame_inst = cv2.cvtColor(frame_inst, cv2.COLOR_BGR2RGB)
            frame_class = cv2.imread(f"{episode_dir}/{episode_name}/0/{frame.replace('seg_inst', 'seg_class')}")
            frame_class = cv2.cvtColor(frame_class, cv2.COLOR_BGR2RGB)
            # get the unique colors in the instance image
            unique_colors = np.unique(frame_inst.reshape(-1, frame_inst.shape[2]), axis=0)
            # for each unique color, get the class color of that region
            for color in unique_colors:
                # get the mask of the object
                mask = np.all(frame_inst == color, axis=-1)
                # ignore if there are less than 20 pixels in the mask
                if mask.sum() < 100:
                    continue
                # get the unique colors in the instance/class images masked
                frame_inst_mode = get_mode(frame_inst[mask])
                inst_color = str(list([int(x) for x in frame_inst_mode]))
                
                frame_class_mode = get_mode(frame_class[mask])
                class_color = str(list([int(x) for x in frame_class_mode]))
                
                # if the instance color is not in the map, it's an ignore object, so continue
                if inst_color not in inst_color_to_candidate_classes:
                    continue

                # if the class color already exists, continue
                if str(class_color) in class_color_to_object_class:
                    continue
                # elif there is only one candidate class, use that
                elif len(inst_color_to_candidate_classes[str(inst_color)]) == 1:
                    obj = inst_color_to_candidate_classes[str(inst_color)][0]
                    print("Only one candidate")
                # otherwise prompt the user to select the candidate class
                else:
                    print("class color", class_color, "all", class_color_to_object_class)
                    print("Frame", frame, "Select the class for the object with color", str(class_color), "from the following options:")
                    for i, obj in enumerate(inst_color_to_candidate_classes[str(inst_color)]):
                        print(i, obj)
                    # display the frame with a red box around the object
                    frame_inst[mask] = [255, 255, 255]
                    cv2.imshow("Select the class for the object", frame_inst)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    class_idx = int(input("Enter the number of the class: "))
                    obj = inst_color_to_candidate_classes[str(inst_color)][class_idx]
                    inst_color_to_candidate_classes[str(inst_color)].pop(class_idx)
                # add the color to the dictionary
                class_color_to_object_class[str(class_color)] = obj
                print("Adding color mapping", str(class_color), obj)

        # save the class colormap pickle file
        with open(f"{episode_dir}/{episode_name}/class_colormap.pkl", "wb") as f:
            pickle.dump(class_color_to_object_class, f)
    
    return class_color_to_object_class, initial_objects