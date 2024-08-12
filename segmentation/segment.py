# segment.py: wrapper for SAM2

import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "../sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# given a bounding box, get the segmentation
# inputs: RGB image, detected bounding boxes
# outputs: segmentation masks (should be 1 per bounding box)
def segment(rgb, boxes):
    # edge case: no boxes, return with no masks
    if len(boxes) == 0:
        return np.array([])
    predictor.set_image(rgb)  # give the image to the predictor
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )  # predict the segmentations using the bounding boxes
    masks = masks.reshape((masks.shape[0], rgb.shape[0], rgb.shape[1]))  # N x 1 x 512 x 512 to N x 512 x 512
    return masks
    