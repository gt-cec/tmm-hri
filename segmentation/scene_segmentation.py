"""
FILE NAME: scene_segmentation
DESCRIPTION: segment and classify images using detectron2 library trained on COCO dataset using CPU only.
INPUT: image (JPG)
OUTPUT: Class ID, bounding box coordinates

"""

import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random, ast, math

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# configure model
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu' # change device from gpu to cpu
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# convert class id to labels
with open(os.path.dirname(os.path.abspath(__file__)) + '/coco_id2label.txt', 'r') as file:
    id2label = file.read()
id2label = ast.literal_eval(id2label)

def segmentation(img):
    print(f'image dimensions: {img.shape}') # image dimensions

    # run segmentation
    output = predictor(img)
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    # Use `Visualizer` to draw the predictions on the image.
    # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.00)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('segmented image', out.get_image()[:, :, ::-1])
    # cv2.waitkey(0)
    # cv2.destroyAllWindows()

    pred_class_labels = [(id + 1, id2label[id + 1]) for id in output["instances"].pred_classes.tolist()] # add 1 to id because of discrepancies in dictionary
    # print labels
    # print(f'class id and labels: {output["instances"].pred_classes},{pred_class_labels}')
    # print(output["instances"].pred_masks.shape)
    return pred_class_labels, output["instances"].pred_masks
