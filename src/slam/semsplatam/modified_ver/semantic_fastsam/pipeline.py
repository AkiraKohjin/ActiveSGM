import torch
import numpy as np
from pycocotools import mask

def postprocess_fastSAM(result, threshold=0.5):
    boxes = result.boxes.data  # get bounding boxes
    masks = result.masks.data  # get masks

    # List to hold post-processed results for each object
    post_processed_results = []

    for i in range(masks.shape[0]):  # Iterate over all detected objects
        post_processed_result = {}

        post_processed_result['bbox'] = boxes[i].tolist()  # convert to list for JSON serialization
        post_processed_result['segmentation'] = {}  # The segmentation info is not directly available in FastSAM
        post_processed_result['segmentation']['size'] = list(result.orig_shape)

        binary_mask = (masks[i] > threshold).type(torch.uint8)
        rle = mask.encode(np.asfortranarray(binary_mask.cpu().numpy()))
        post_processed_result['segmentation']['counts'] = rle['counts']
        post_processed_result['area'] = mask.area(rle)

        post_processed_result['predicted_iou'] = 0  # placeholder
        post_processed_result['point_coords'] = [0, 0]  # placeholder
        post_processed_result['stability_score'] = 0  # placeholder

        post_processed_result['crop_box'] = [0, 0, result.orig_shape[1], result.orig_shape[0]]
        post_processed_results.append(post_processed_result)

    return post_processed_results



