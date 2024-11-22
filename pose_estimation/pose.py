# pose.py: conducts 2D pose detection and lifting to 3D

# get the forward direction of a character
# pose: list of keypoints
def get_direction_from_pose(pose:list, use_gt_pose=False) -> list:
    # get direction from ground truth pose
    if use_gt_pose:
        return pose[-1]  # use the coordinate system (east, north, vertical)
    # get direction from observed pose
    else:
        print("POSE DIRECTION FROM NON-GT IS NOT IMPLEMENTED")
        return pose[-1]  # use the coordinate system (east, north, vertical)
