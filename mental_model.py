# mental_model.py: uses the segmentator and the dynamic scene graph to construct and update the mental model

from dsg import dsg
from segmentation import scene_segmentation
import math

class MentalModel:
    def __init__(self):
        self.dsg = dsg.DSG()
        self.fov = 1.0472  # 60deg in radians

    # updates the DSG from a known pose and RGBD image
    # Coordinate Frame: x (right), y (up), z (forward)
    def update_from_rgbd_and_pose(self, rgb, depth, pose):
        # verify types
        # segment the RGB into labels and masks
        labels, seg_masks = scene_segmentation.segmentation(rgb)
        # overlay the object masks on the depth image
        objects = []  # resulting objects and their coordinates
        for i in range(seg_masks.shape[0]):  # for each observed object
            mask = depth[seg_masks[i]]  # the depth field corresponding to the mask
            dist = mask.sum() / seg_masks[i].sum()  # mean depth
            indices = seg_masks[i].nonzero()  # get indices of the mask
            avg_row = indices[:, 0].float().mean()  # get average row
            avg_col = indices[:, 1].float().mean()  # get average col
            horz_angle = (avg_col - seg_masks[i].shape[1] / 2) / seg_masks[i].shape[1] * self.fov  # get angle left/right from center
            vert_angle = (avg_row - seg_masks[i].shape[0] / 2) / seg_masks[i].shape[0] * self.fov  # get angle up/down from center
            x_pos_local = math.sin(horz_angle) * dist
            y_pos_local = math.sin(vert_angle) * dist
            direction = pose[3]  # forward vector
            pose_horz_angle = math.atan2(direction[2], direction[0])  # for the pose, z is horz plan up (y), x is horz plan right (x)
            pose_vert_angle = math.asin(direction[1])  # for the pose, y is the vert plan up, so angle is y / 1
            x_pos_global = pose[0][0] + math.sin(pose_horz_angle + horz_angle) * dist
            y_pos_global = pose[0][1] + math.sin(pose_vert_angle + vert_angle) * dist  # should use head instead fir eye vert
            z_pos_global = pose[0][2] + math.sin(pose_horz_angle + horz_angle) * dist
            # print(labels[i], "dist", dist, avg_row, avg_col, "angle", horz_angle, vert_angle, "pos", x_pos_local, y_pos_local, "pose angle", pose_horz_angle, pose_vert_angle, "global", x_pos_global, y_pos_global, z_pos_global)
            objects.append({
                "class": labels[i][1],
                "class id": labels[i][0],
                "x": float(x_pos_global),
                "y": float(y_pos_global),
                "z": float(z_pos_global),
                "seg mask": seg_masks[i]
            })

        return objects
