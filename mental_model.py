# mental_model.py: uses the segmentator and the dynamic scene graph to construct and update the mental model

from dsg import dsg
# from segmentation import scene_segmentation
from detection import detect
from segmentation import segment
import math, numpy

class MentalModel:
    def __init__(self):
        self.dsg = dsg.DSG()
        self.fov = 1.0472  # 60deg in radians

    # updates the DSG from a known pose and RGBD image
    # Coordinate Frame: x (right), y (up), z (forward)
    def update_from_rgbd_and_pose(self, rgb, depth, pose, classes, depth_classes=[], seg_threshold=0.1, seg_save_name=None, depth_test=None):
        # verify types
        # get objects in the scene
        detected_objects, rgb_with_boxes = detect.detect(rgb, classes, seg_threshold, seg_save_name)

        # get humans in the scene
        detected_humans, _ = detect.detect(rgb, depth_classes, seg_threshold, seg_save_name)
        detected_objects += detected_humans

        # get humans figures using depth, this is used to double check the humans
        depth_3channel = numpy.expand_dims(depth, axis=0)  # Shape becomes (1, 2, 2)
        depth_3channel = numpy.tile(depth_3channel, (3, 1, 1))  # Shape becomes (3, 2, 2)
        depth_detected_objects, depth_with_boxes = detect.detect(depth_3channel * 20, depth_classes, 0.4, None)
        human_class_id = [i for i, x in enumerate(classes) if x == "human"][0]  # get the class ID of the "human" label, this could be optimized a little by placing this ID as a class level variable
        for i in range(len(depth_detected_objects)):  # set all outputs to human
            depth_detected_objects[i]["class"] = "human"
            depth_detected_objects[i]["class id"] = human_class_id
        
        print("DEPTH", depth_detected_objects)
        
        # remove the humans from the detected objects that do not overlap with the depth 
        print("DETECTED BEFORE", detected_objects)
        detected_objects = detect.remove_objects_not_overlapping(detected_objects, depth_detected_objects, overlap_threshold=0.8, classes_to_filter=["human"])
        print("DETECTED AFTER", detected_objects)

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
            z_pos_local = math.sin(vert_angle) * dist
            direction = pose[3]  # forward vector
            pose_horz_angle = math.atan2(direction[2], direction[0])  # for the pose, z is horz plan up (y), x is horz plan right (x)
            pose_vert_angle = math.asin(direction[1])  # for the pose, y is the vert plan up, so angle is y / 1
            x_pos_global = pose[0][0] + math.cos(pose_horz_angle - horz_angle) * dist
            y_pos_global = pose[0][2] + math.sin(pose_horz_angle - horz_angle) * dist
            z_pos_global = pose[0][1] + math.sin(pose_vert_angle - vert_angle) * dist  # in the future, should use head instead for eye vert
            detected_objects[i]["x"] = float(x_pos_global)
            detected_objects[i]["y"] = float(y_pos_global)
            detected_objects[i]["z"] = float(z_pos_global)
            detected_objects[i]["seg mask"] = seg_masks[i]
            detected_objects[i]["debug"] = f"class {detected_objects[i]["class"]} dist {dist} horz angle {horz_angle} x local {x_pos_local} horz angle global {pose_horz_angle + horz_angle} x global {x_pos_global} y global {y_pos_global}"
            
            print("OBJ at ", detected_objects[i]["debug"])
            # input()

        return detected_objects
