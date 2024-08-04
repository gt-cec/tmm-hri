# mental_model.py: uses the segmentator and the dynamic scene graph to construct and update the mental model

from dsg import dsg
from segmentation import scene_segmentation
import math

class MentalModel:
    def __init__(self):
        self.dsg = dsg.DSG()
        self.fov = 1.0472  # 60deg in radians

    # updates the DSG from a known pose and RGBD image
    def update_from_rgbd_and_pose(self, rgb, depth, pose):
        # verify types
        # segment the RGB into labels and masks
        labels, seg_masks = scene_segmentation.segmentation(rgb)
        # overlay the object masks on the depth image
        for i in range(seg_masks.shape[0]):  # for each observed object
            mask = depth[seg_masks[i]]  # the depth field corresponding to the mask
            dist = mask.sum() / seg_masks[i].sum()  # mean depth
            indices = seg_masks[i].nonzero()  # get indices of the mask
            avg_row = indices[:, 0].float().mean()  # get average row
            avg_col = indices[:, 1].float().mean()  # get average col
            horz_angle = (avg_col - seg_masks[i].shape[1] / 2) / seg_masks[i].shape[1] * self.fov
            vert_angle = (avg_row - seg_masks[i].shape[0] / 2) / seg_masks[i].shape[0] * self.fov
            x_pos_local = math.sin(horz_angle) * dist
            y_pos_local = math.sin(vert_angle) * dist
            print(pose)
            print(labels[i], "dist", dist, avg_row, avg_col, "angle", horz_angle, vert_angle, "pos", x_pos_local, y_pos_local)
        # project from the pose

        # update objects on the DSG
