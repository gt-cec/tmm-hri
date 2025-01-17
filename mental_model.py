# mental_model.py: uses the segmentator and the dynamic scene graph to construct and update the mental model

from dsg import dsg
from detection import detect
from segmentation import segment
from pose_estimation import pose
import math, numpy, utils
import cv2

class MentalModel:
    def __init__(self, pose_detector=None):
        self.dsg = dsg.DSG()
        self.fov = 40
        self.pose_detector = pose.PoseDetection() if pose_detector is None else pose_detector

    # initializes the DSG from a list of objects
    def initialize(self, objects:list, verbose=False) -> None:
        self.dsg.initialize_scene(objects)  # pass to the DSG
        return

    # updates the DSG from a known pose and RGBD image
    # Coordinate Frame: x (right), y (forward), z (up)
    def update_from_rgbd_and_pose(self, rgb, depth, pose, classes, class_to_class_id=[], depth_classes=[], gt_semantic=None, gt_instance_colormap=None, seg_threshold=0.1, seg_save_name=None, depth_test=None):
        # verify types
        human_class_id = [i for i, x in enumerate(classes) if x == "human"][0]  # get the class ID of the "human" label, this could be optimized a little by placing this ID as a class level variable

        # check if given GT semantics
        assert gt_semantic is None or gt_semantic is not None and gt_instance_colormap is not None, "A ground truth image was provided, however a colormap was not! Please include a colormap."

        have_gt_detections = gt_semantic is not None and gt_instance_colormap is not None  # flag for if we have the ground truth detections
        detected_humans, depth_detected_humans, filtered_detected_humans = [], [], []  # initialize these variables as they are returned

        # if we have GT detections, parse them into the objects
        if have_gt_detections:
            print("    Using object detection from ground truth information!")
            detected_objects, object_seg_masks = detect.detect_from_ground_truth(gt_semantic, gt_instance_colormap, classes=classes, class_to_class_id=class_to_class_id)
            rgb_detected_humans = None
            depth_detected_humans = None
            for i in range(len(detected_objects)):
                detected_objects[i]["seg mask"] = object_seg_masks[i,:,:] == 1
        # if we do not have GT detections, run detection through the RGB
        else:
            # get objects in the scene
            detected_objects, _ = detect.detect(rgb, classes, seg_threshold, seg_save_name)  # returns detected objects and an RGB debugging image (ignored)

            # get humans in the scene
            rgb_detected_humans, _ = detect.detect(rgb, depth_classes, 0.1, None)
            for i in range(len(rgb_detected_humans)):
                rgb_detected_humans[i]["class"] = "human"
                rgb_detected_humans[i]["class id"] = human_class_id

            # get humans figures using depth, this is used to double check the humans
            depth_3channel = numpy.tile(numpy.expand_dims(depth, axis=0), (3, 1, 1))  # Shape becomes (3, 2, 2)
            depth_detected_humans, depth_with_boxes = detect.detect(depth_3channel * 20, depth_classes, 0.4, None)  # multiplying depth by *20 makes it a more contrastive image
            for i in range(len(depth_detected_humans)):  # set all outputs to human
                depth_detected_humans[i]["class"] = "human"
                depth_detected_humans[i]["class id"] = human_class_id

            # remove the humans from the detected humans that do not overlap with the depth
            detected_humans = detect.remove_objects_not_overlapping(rgb_detected_humans, depth_detected_humans, overlap_threshold=0.3, classes_to_filter=["human"])
            
            # segment the objects
            object_boxes = [o["box"] for o in detected_objects]
            human_boxes = [o["box"] for o in detected_humans]
            seg_masks = segment.segment(rgb, object_boxes + human_boxes)
            object_seg_masks = seg_masks[:len(object_boxes)]
            human_seg_masks = seg_masks[len(object_boxes):]
            for i in range(human_seg_masks.shape[0]):
                detected_humans[i]["seg mask"] = human_seg_masks[i,:,:]
            for i in range(object_seg_masks.shape[0]):
                detected_objects[i]["seg mask"] = object_seg_masks[i,:,:]

        # get the pose for each human
        detected_humans = self.get_human_poses_from_rgb_seg_depth_and_detected_humans(rgb, depth, detected_humans, pose)

        # make sure the seg mask and detected object dimensions match
        assert len(detected_objects) == len(object_seg_masks), f"The number of detected objects ({len(detected_objects)}) does not equal the number of the segmentation masks ({len(object_seg_masks)}), one of these modules is misperforming."
        
        # project object locations to the robot's pose
        detected_objects = utils.project_detected_objects_positions_given_seg_masks_and_agent_pose(detected_objects, pose, object_seg_masks, depth, self.fov)
        
        # update the dynamic scene graph
        self.dsg.update(detected_objects)

        return detected_objects, (detected_humans, rgb_detected_humans, depth_detected_humans)

    # get human poses
    def get_human_poses_from_rgb_seg_depth_and_detected_humans(self, rgb, depth_map, detected_humans, robot_pose, frame="0"):
        if len(detected_humans) == 0:
            return detected_humans
    
        predicted_location, predicted_heading, (pred_human_relative_loc, mean_depth, first_person_3d, keypoints, pred_skeleton) = self.pose_detector.get_heading_of_person(rgb, depth_map, detected_humans, robot_pose)
        detected_humans[0]["pose"] = predicted_location
        detected_humans[0]["direction"] = predicted_heading
        detected_humans[0]["keypoints"] = keypoints[0]
        return detected_humans

    # if you already have the detected objects through preprocessing, just update the DSG directly
    def update_from_detected_objects(self, detected_objects):
        self.dsg.update(detected_objects)
        return

    # get the objects that should be visible from a given location and pose
    def get_objects_in_visible_region(self, location, direction):
        fov = math.radians(self.fov) / 2  # convert to radians, divide by two because angles will be calculated from the center of the field of view
        visible_objects = []
        for object_id in self.dsg.objects:
            # get the object location relative to the agent location
            object_location_wrt_agent_location = numpy.array([self.dsg.objects[object_id]["x"] - location[0], self.dsg.objects[object_id]["y"] - location[1], self.dsg.objects[object_id]["z"] - location[2]])
            # check if object is in the agent's field of view
            # NOTE: will need to filter by walls/visibility
            angle = math.acos(numpy.dot(object_location_wrt_agent_location, direction) / (numpy.linalg.norm(object_location_wrt_agent_location) * numpy.linalg.norm(direction)))
            if angle < fov:
                visible_objects.append(self.dsg.objects[object_id].as_dict())
        return visible_objects
