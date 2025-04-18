# detect.py: conducts object detection to get the bounding boxes of relevant objects in the scene

import numpy as np
import utils


# gets the closest match to a color in the ground truth instance map, used for VirtualHome's simulator ground truth
def get_closest_match(self, color, gt_instances_color_map):
    closest_match = None
    closest_match_dist = 1000000
    dists = []
    for k in gt_instances_color_map:
        instance_color = k[0]
        instance_class = k[1]
        dist = ((color[0] - instance_color[0]) ** 2 + (color[1] - instance_color[1]) ** 2 + (color[2] - instance_color[2]) ** 2) ** 0.5
        if dist < closest_match_dist:
            closest_match = instance_color
            closest_match_dist = dist
        dists.append((dist, instance_class))
    return closest_match, closest_match_dist, sorted(dists)


# processes a ground truth instance map and color map to get objects, can filter in classes
def detect_from_ground_truth(self, gt_instances_image, gt_class_image, gt_class_colormap, classes=[], class_to_class_id=[]):
    objects = []  # objects detected
    segments = np.empty(gt_instances_image.shape[:2])
    segments = segments[np.newaxis, ...]
    unique_colors = np.unique(gt_instances_image.reshape(-1, gt_instances_image.shape[-1]), axis=0)  # unique colors in the image
    for color_unique in unique_colors:
        # mask the instance
        mask = np.all(gt_instances_image[:,:] == color_unique, axis=-1)
        # get the color of the class image at the mask
        class_color = [int(x) for x in utils.get_mode(gt_class_image[mask])]
        color_key = str(class_color)
        if color_key not in gt_class_colormap:  # catch colors not in the ground truth colors
            closest_match, closest_match_dist, dists = get_closest_match(eval(color_key), [(eval(x), gt_class_colormap[x]) for x in gt_class_colormap])
            if closest_match_dist < 10:
                color_key = str(list(closest_match))
            else:
                continue
        if classes != [] and gt_class_colormap[color_key] not in classes:  # catch classes that are not in the filter in list
            continue
        segment_2d = np.all(gt_instances_image[:,:] == color_unique, axis=-1)
        segment = segment_2d[np.newaxis, ...]
        true_indices = np.argwhere(segment_2d)
        if len(true_indices) != 0:  # if the instance was segmented
            min_coords = true_indices.min(axis=0)
            max_coords = true_indices.max(axis=0)
            box = np.array((min_coords, max_coords))
            segments = np.concatenate((segments, segment), axis=0)
            obj_class = gt_class_colormap[color_key]
            objects.append({
                "class": obj_class,
                "class id": class_to_class_id[obj_class] if class_to_class_id != [] else classes.index(obj_class) if classes != [] else -1,
                "confidence": 1,  # perfect confidence from simulator
                "box": box,
                "center": [(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2]
            })
    segments = segments[1:,:,:]
    return objects, segments