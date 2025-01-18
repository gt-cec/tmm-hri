# detect.py: conducts object detection to get the bounding boxes of relevant objects in the scene

import PIL
import torch
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os
import sys

# set device to cuda if cuda is available
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")
# otherwise check if on macos
elif sys.platform == "darwin":
    device = "mps"
    print("Using MPS")
else:
    device = "cpu"
    print("Using CPU")

# if you get an undefined symbol:ffi_type_uint32, version LIBFFI_BASE_7.0 error, set the env var LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(torch.device(device))

def detect(image, classes, threshold=0.1, save_name=None):
    texts = [["" + c for c in classes]]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(torch.device(device))
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = inputs.pixel_values.shape[-2:]

    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=[(h, w)], threshold=threshold
    )[0]  # we only pass one image in, so can take the first result [[results]]

    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]

    clip_to_orig_width, clip_to_orig_height = image.shape[1] / w, image.shape[0] / (h * (image.shape[0] / image.shape[1]))
    objects = []
    object_idx_by_class_id = {}  # for checking overlapping boxes
    class_list = list(classes)
    for box, score, label, i in zip(boxes, scores, labels, range(len(boxes))):
        box = [round(i, 2) for i in box.tolist()]
        box[0] *= clip_to_orig_width
        box[1] *= clip_to_orig_height
        box[2] *= clip_to_orig_width
        box[3] *= clip_to_orig_height
        box = [[int(box[0]), int(box[1])], [int(box[2]), int(box[3])]]
        label = int(label)
        objects.append({
            "class": class_list[label],
            "class id": label,
            "confidence": round(score.item(), 3),
            "box": box,  # [[x1,y1], [x2,y2]] from top left, NOT [[row1,col1],[row2,col2]]
            "center": [(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2]
        })
        if label not in object_idx_by_class_id:
            object_idx_by_class_id[label] = [i]
        else:
            object_idx_by_class_id[label].append(i)
        print(f"Detected {texts[0][label]} {label} with confidence {round(score.item(), 3)} at location {box}")

    # remove objects that significantly overlap by choosing highest
    overlap_threshold = 0.9
    for class_id in object_idx_by_class_id:
        for object_idx_1 in range(len(object_idx_by_class_id[class_id])):
            for object_idx_2 in range(object_idx_1, len(object_idx_by_class_id[class_id])):
                # skip if same index or we have already thrown out one of the objects
                if object_idx_1 == object_idx_2 or objects[object_idx_by_class_id[class_id][object_idx_2]] is None or objects[object_idx_by_class_id[class_id][object_idx_1]] is None:
                    continue
                if calculate_overlap_proportion(objects[object_idx_by_class_id[class_id][object_idx_1]]["box"], objects[object_idx_by_class_id[class_id][object_idx_2]]["box"]) > overlap_threshold:
                    if objects[object_idx_by_class_id[class_id][object_idx_1]]["confidence"] > objects[object_idx_by_class_id[class_id][object_idx_2]]["confidence"]:
                        objects[object_idx_by_class_id[class_id][object_idx_2]] = None
                    else:
                        objects[object_idx_by_class_id[class_id][object_idx_1]] = None
    objects = [x for x in objects if x is not None]  # remove the objects we filtered out

    if False and save_name is not None:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        for o in objects:
            x1, y1, x2, y2 = o["box"]
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            draw.text(xy=(x1, y1), text=texts[0][label])

        image.save(save_name + ".png")

    return objects, image

def get_closest_match(color, gt_instances_color_map):
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
def detect_from_ground_truth(gt_instances_image, gt_instances_color_map, classes=[], class_to_class_id=[]):
    objects = []  # objects detected
    segments = np.empty(gt_instances_image.shape[:2])
    segments = segments[np.newaxis, ...]
    unique_colors = np.unique(gt_instances_image.reshape(-1, gt_instances_image.shape[-1]), axis=0)  # unique colors in the image
    for color_unique in unique_colors:
        color_key = str(list(color_unique))
        # print("##", color_unique_reformatted, gt_instances_color_map)
        if color_key not in gt_instances_color_map:  # catch colors not in the ground truth colors
            closest_match, closest_match_dist, dists = get_closest_match(eval(color_key), [(eval(x), gt_instances_color_map[x]["class"]) for x in gt_instances_color_map])
            if closest_match_dist < 2:
                print("MATCHED", color_key, closest_match, closest_match_dist, gt_instances_color_map[str(closest_match)], dists)
                color_key = str(list(closest_match))
            else:
                print("MISSING", color_key, closest_match, closest_match_dist)
                continue
        if classes != [] and gt_instances_color_map[color_key]["class"] not in classes:  # catch classes that are not in the filter in list
            continue
        print("DETECTED", gt_instances_color_map[color_key]["class"])
        segment_2d = np.all(gt_instances_image[:,:] == color_unique, axis=-1)
        segment = segment_2d[np.newaxis, ...]
        true_indices = np.argwhere(segment_2d)
        if len(true_indices) != 0:  # if the instance was segmented
            min_coords = true_indices.min(axis=0)
            max_coords = true_indices.max(axis=0)
            box = np.array((min_coords, max_coords))
            segments = np.concatenate((segments, segment), axis=0)
            obj_class = gt_instances_color_map[color_key]["class"]
            objects.append({
                "class": obj_class,
                "class id": class_to_class_id[obj_class] if class_to_class_id != [] else classes.index(obj_class) if classes != [] else -1,
                "confidence": 1,  # perfect confidence from simulator
                "box": box,
                "center": [(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2]
            })
    segments = segments[1:,:,:]
    return objects, segments

# removes boxes from detected object set A (main set) that are not in detected object set B (check set)
# this is used for syncing RGB and depth detected objects
def remove_objects_not_overlapping(objects_main, objects_check, overlap_threshold=0.8, classes_to_filter=None):
    classes = {}  # organize by class
    for i, o in enumerate(objects_main):  # set up classes for the main objects
        if classes_to_filter is None or o["class"] not in classes_to_filter:  # if classes to filter were specified, don't count classes that were not specified
            continue
        if o["class"] not in classes:
            classes[o["class"]] = [[], []]  # 0th index: main, 1st index: check
        classes[o["class"]][0].append(i)  # add the object to this class
    for i, o in enumerate(objects_check):  # set up classes for the check objects
        if o["class"] not in classes:  # ignore check objects not in the main objects
            continue
        classes[o["class"]][1].append(i)  # add the object to this class
    main_objects_to_remove = []
    for c in classes:  # remove overlapping
        for obj_main_idx in classes[c][0]:  # for each main object in a class
            closest_match_check_classes_idx = None
            closest_match_check_overlap = 0
            for classes_check_idx, obj_check_idx in enumerate(classes[c][1]):  # for each check object of that class
                overlap = calculate_overlap_proportion(objects_main[obj_main_idx]["box"], objects_check[obj_check_idx]["box"])
                if overlap >= overlap_threshold and overlap > closest_match_check_overlap:  # check if the overlap passes the threshold and is more than other overlaps
                    closest_match_check_classes_idx = classes_check_idx
                    closest_match_check_overlap = overlap
            if closest_match_check_classes_idx is None:  # if main object has no check objects, it is not overlapping anything, so delete the main object
                main_objects_to_remove.append(obj_main_idx)
            else:  # otherwise, there is a close match to a check object, so keep the main object and delete the check object from the classes
                del classes[c][1][closest_match_check_classes_idx]
    return [x for i, x in enumerate(objects_main) if i not in main_objects_to_remove]  # return the updated objects main

# gets the percent overlap of two boxes, generated with Claude
def calculate_overlap_proportion(box1, box2):
    # unpack the coordinates
    ((x1_1, y1_1), (x2_1, y2_1)) = box1
    ((x1_2, y1_2), (x2_2, y2_2)) = box2

    # calculate the coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:  # check if there is an overlap
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)  # calculate the area of intersection

    # calculate the area of both boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - intersection_area  # calculate the union area
    overlap_proportion = intersection_area / union_area  # calculate the overlap proportion
    return overlap_proportion
