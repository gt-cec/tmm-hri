# mental_model.py: uses the segmentator and the dynamic scene graph to construct and update the mental model

from dsg import dsg
# from segmentation import scene_segmentation
from detection import detect
from segmentation import segment
import math

class MentalModel:
    def __init__(self):
        self.dsg = dsg.DSG()
        self.fov = 1.0472  # 60deg in radians

    # updates the DSG from a known pose and RGBD image
    # Coordinate Frame: x (right), y (up), z (forward)
    def update_from_rgbd_and_pose(self, rgb, depth, pose, classes, seg_threshold=0.1, seg_save_name=None):
        # verify types
        # get objects in the scene
        detected_objects, rgb_with_boxes = detect.detect(rgb, classes, seg_threshold, seg_save_name)
        # segment the objects
        boxes = [o["box"] for o in detected_objects]
        seg_masks = segment.segment(rgb, boxes)
            
        # make sure the seg mask and detected object dimensions match
        assert len(detected_objects) == len(seg_masks), f"The number of detected objects ({len(detected_objects)}) does not equal the number of the segmentation masks ({len(seg_masks)}), one of these modules is misperforming."
        
        # get the location of the object in 3D space
        for i in range(seg_masks.shape[0]):  # for each observed object
            mask = depth[seg_masks[i] > 0]  # the depth field corresponding to the object mask
            dist = mask.sum() / seg_masks[i].sum()  # mean depth
            indices = seg_masks[i].nonzero()  # get indices of the mask
            avg_row = sum(indices[0]) / len(indices[0])  # get average row
            avg_col = sum(indices[1]) / len(indices[1])  # get average col
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
            detected_objects[i]["x"] = float(x_pos_global)
            detected_objects[i]["y"] = float(y_pos_global)
            detected_objects[i]["z"] = float(z_pos_global)
            detected_objects[i]["seg mask"] = seg_masks[i]
            
            # print("OBJ at ", x_pos_global, y_pos_global, z_pos_global, "user at", pose[0][0], pose[0][1], pose[0][2], "user dir", direction)
            # input()

        return detected_objects
