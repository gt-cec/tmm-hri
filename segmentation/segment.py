# segment.py: wrapper for SAM2

import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import hydra
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # adding SAM2 config path from here: https://github.com/facebookresearch/sam2/issues/81#issuecomment-2262979343
    # hydra is initialized on import of sam2, which sets the search path which can't be modified
    # so we need to clear the hydra instance
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # reinit hydra with a new search path for configs
    # hydra.initialize_config_module('', version_base='1.2')/Users/jack/Documents/Installations/sam2

    # this should work now
    # model = build_sam2('<config-name>', '<checkpoint-path>')

    sam2_checkpoint = "/Users/jack/Documents/Installations/sam2/sam2/configs/sam2.1/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="mps")
    predictor = SAM2ImagePredictor(sam2_model)
except ModuleNotFoundError as e:
    print("WARNING: SAM2 is not installed, do not expect to use it. Message:", e)
    predictor = None

# given a bounding box, get the segmentation
# inputs: RGB image, detected bounding boxes
# outputs: segmentation masks (should be 1 per bounding box)
def segment(rgb, boxes):
    """
    Given an RGB image and a list of bounding boxes, run a model to segment the bounding boxes. Uses SAM2.

    Inputs:
        rgb: Image in typical OpenCV RGB format: numpy array (H,W,3)
        boxes: List of N bounding boxes: [[[col1, row1], [col2, row2]], ... ]

    Output:
        masks: segmentation masks for each of the N boxes stacked: numpy array (N,H,W)
    """
    # edge case: no boxes, return with no masks
    if len(boxes) == 0:
        return np.array([])
    # check if predictor is None (mean's SAM2 is not installed)
    assert predictor is not None, "The segmentation system is being called, however SAM2 is not installed. Install SAM2 first."

    predictor.set_image(rgb)  # give the image to the predictor
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=False,
    )  # predict the segmentations using the bounding boxes
    masks = masks.reshape((masks.shape[0], rgb.shape[0], rgb.shape[1]))  # N x 1 x 512 x 512 to N x 512 x 512
    return masks == 1  # convert from 0's and 1's to False and True
