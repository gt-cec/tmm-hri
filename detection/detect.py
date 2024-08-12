# detect.py: conducts object detection to get the bounding boxes of relevant objects in the scene

import requests
from PIL import Image, ImageDraw
import torch
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# if you get an undefined symbol:ffi_type_uint32, version LIBFFI_BASE_7.0 error, set the env var LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(torch.device("cuda"))

def detect(image, classes, threshold=0.1, save_name=None):
    texts = [["" + c for c in classes]]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(torch.device("cuda"))
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
    for box, score, label, i in zip(boxes, scores, labels, range(len(boxes))):
        box = [round(i, 2) for i in box.tolist()]
        box[0] *= clip_to_orig_width
        box[1] *= clip_to_orig_height
        box[2] *= clip_to_orig_width
        box[3] *= clip_to_orig_height
        label = int(label)
        objects.append({
            "class": classes[label], 
            "class id": label, 
            "confidence": round(score.item(), 3), 
            "box": box, 
            "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
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
                if object_idx_1 == object_idx_2:
                    continue
                if calculate_overlap_percentage(objects[object_idx_by_class_id[class_id][object_idx_1]]["box"], objects[object_idx_by_class_id[class_id][object_idx_2]]["box"]) > overlap_threshold:
                    print("OVERLAPPPP")
                    if objects[object_idx_by_class_id[class_id][object_idx_1]]["confidence"] > objects[object_idx_by_class_id[class_id][object_idx_2]]["confidence"]:
                        objects[object_idx_by_class_id[class_id][object_idx_2]] = None
                    else:
                        objects[object_idx_by_class_id[class_id][object_idx_1]] = None
    objects = [x for x in objects if x is not None]
                    

    if save_name is not None:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        
        for o in objects:
            x1, y1, x2, y2 = o["box"]
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            draw.text(xy=(x1, y1), text=texts[0][label])
        
        image.save(save_name + ".png")

    return objects, image

# gets the percent overlap of two boxes, generated with Claude
def calculate_overlap_percentage(box1, box2):
    # unpack the coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
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
    overlap_percentage = (intersection_area / union_area) * 100  # calculate the overlap percentage
    return overlap_percentage