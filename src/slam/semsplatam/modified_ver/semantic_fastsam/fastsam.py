import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as f
from PIL import Image
import torchvision.transforms as transforms
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from third_parties.Semantic_Fast_SAM.configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from third_parties.Semantic_Fast_SAM.configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from third_parties.Semantic_Fast_SAM.clipseg import clipseg_segmentation
from third_parties.Semantic_Fast_SAM.oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation, oneformer_cityscapes_segmentation

oneformer_func = {
    'ade20k': oneformer_ade20k_segmentation,
    'coco': oneformer_coco_segmentation,
}

from pycocotools import mask


def clip_classification(image, class_list, clip_processor, clip_model, rank):
    inputs = clip_processor(text=class_list, images=image, return_tensors="pt", padding=True).to(rank)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    class_name = class_list[probs.argmax().item()]
    return class_name, probs
    # if top_k == 1:
    #     class_name = class_list[probs.argmax().item()]
    #     return class_name
    # else:
    #     top_k_indices = probs.topk(top_k, dim=1).indices[0]
    #     top_k_class_names = [class_list[index] for index in top_k_indices]
    #     return top_k_class_names


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


def load_filename_with_extensions(data_path, filename):
    """
    Returns file with corresponding extension to json file.
    Raise error if such file is not found.

    Args:
        filename (str): Filename (without extension).

    Returns:
        filename with the right extension.
    """
    full_file_path = os.path.join(data_path, filename)
    # List of image file extensions to attempt
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    # Iterate through image file extensions and attempt to upload the file
    for ext in image_extensions:
        # Check if the file with current extension exists
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext  # Return True if file is successfully uploaded
    raise FileNotFoundError(f"No such file {full_file_path}, checked for the following extensions {image_extensions}")


def get_bbox_from_mask(mask: torch.Tensor):
    """
    Extract the bounding box (x1, y1, x2, y2) from a boolean mask.

    Args:
        mask (torch.Tensor): A (H, W) boolean tensor.

    Returns:
        list: [x1, y1, x2, y2] representing the bounding box.
    """
    # Get coordinates of non-zero (True) values
    coords = mask.nonzero()

    # Find bounding box limits
    y1, x1 = coords.amin(dim=0)  # Top-left corner
    y2, x2 = coords.amax(dim=0)  # Bottom-right corner

    return [x1.item(), y1.item(), x2.item(), y2.item()]


def semantic_annotation_pipeline(image, target_class_list, rank, save_img=False, scale_small=1.2,
                                 scale_large=1.6, scale_huge=1.6,
                                 fastsam_model=None,
                                 clip_processor=None,
                                 clip_model=None,
                                 oneformer_ade20k_processor=None,
                                 oneformer_ade20k_model=None,
                                 oneformer_coco_processor=None,
                                 oneformer_coco_model=None,
                                 blip_processor=None,
                                 blip_model=None,
                                 clipseg_processor=None,
                                 clipseg_model=None,
                                 ):

    h,w, _ = image.shape
    n_cls = len(target_class_list)
    tensor_input = image * 255   # (H,W,C) torch.tensor
    tensor_input = tensor_input.to(dtype=torch.uint8)
    breakpoint()
    img_np = tensor_input.clone().cpu().numpy()
    # to_pil = transforms.ToPILImage()
    input = f.to_pil_image(tensor_input.permute(2,0,1))

    mask_result = fastsam_model(
        input,
        device=rank,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9
    )

    img = img_np[..., ::-1]
    anns = {'annotations': postprocess_fastSAM(mask_result[0])}
    bitmasks, class_names = [], []
    breakpoint()
    class_ids_from_oneformer_coco = oneformer_coco_segmentation(Image.fromarray(img), oneformer_coco_processor,
                                                                oneformer_coco_model, rank)
    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(Image.fromarray(img), oneformer_ade20k_processor,
                                                                    oneformer_ade20k_model, rank)

    final_logits = torch.zeros((h,w,n_cls)).to(rank)
    for ann in anns['annotations']:
        # mask_np = mask_np.cpu().numpy()
        # Convert the NumPy array to bytes
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        crop_bbox = get_bbox_from_mask(valid_mask)
        patch_small = mmcv.imcrop(img, np.array(
            [crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]]))
        patch_valid_mask = mmcv.imcrop(valid_mask.numpy(), np.array(
            [crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]]))
        mmcv.imwrite(patch_small, "./patch_small.jpg")
        mmcv.imwrite(255*patch_valid_mask.astype(np.uint8), "./patch_small_mask.jpg")
        breakpoint()

        # get the class ids of the valid pixels
        coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
        ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
        top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
        top_k_ade20k_propose_classes_ids = torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
        local_class_names = set()
        local_class_names = set.union(local_class_names,
                                      set([CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for class_id in
                                           top_k_ade20k_propose_classes_ids]))
        local_class_names = set.union(local_class_names, set(([
            CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in
            top_k_coco_propose_classes_ids])))

        patch_small = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_small)
        patch_large = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_large)
        patch_huge = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                 scale=scale_huge)
        valid_mask_huge_crop = mmcv.imcrop(valid_mask.numpy(), np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                           scale=scale_huge)
        # mmcv.imshow(patch_small)
        # mmcv.imshow(patch_large)
        #op_class_list = open_vocabulary_classification_blip(patch_large, blip_processor, blip_model, rank)
        #local_class_list = list(set.union(local_class_names, set(op_class_list)))  # , set(refined_imagenet_class_names)
        local_class_list = list(set.union(local_class_names))
        breakpoint()
        mask_categories, probs = clip_classification(patch_small, local_class_list,
                                              3 if len(local_class_list) > 3 else len(local_class_list), clip_processor,
                                              clip_model, rank)
        breakpoint()
        class_ids_patch_huge = clipseg_segmentation(patch_large, mask_categories, clipseg_processor, clipseg_model,
                                                    rank).argmax(0)
        breakpoint()
        valid_mask_huge_crop = torch.tensor(valid_mask_huge_crop)
        if valid_mask_huge_crop.shape != class_ids_patch_huge.shape:
            valid_mask_huge_crop = F.interpolate(
                valid_mask_huge_crop.unsqueeze(0).unsqueeze(0).float(),
                size=(class_ids_patch_huge.shape[-2], class_ids_patch_huge.shape[-1]),
                mode='nearest').squeeze(0).squeeze(0).bool()
        top_1_patch_huge = torch.bincount(class_ids_patch_huge[valid_mask_huge_crop].flatten()).topk(1).indices
        top_1_mask_category = mask_categories[top_1_patch_huge.item()]
        breakpoint()

        # ann['class_proposals'] = mask_categories
        ann['class_name'] = str(top_1_mask_category)
        ann['class_proposals'] = mask_categories
        class_names.append(str(top_1_mask_category))
        bitmasks.append(maskUtils.decode(ann['segmentation']))
        # Delete variables that are no longer needed
        del coco_propose_classes_ids
        del ade20k_propose_classes_ids
        del top_k_coco_propose_classes_ids
        del top_k_ade20k_propose_classes_ids
        del patch_small
        del patch_large
        del patch_huge
        del valid_mask_huge_crop
        del op_class_list
        del mask_categories
        del class_ids_patch_huge

    ################################################################
    # mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
    # print('[Save] save SSA-engine annotation results: ', os.path.join(output_path, filename + '_semantic.json'))
    # valid_mask = torch.tensor(maskUtils.decode({'size': list(ann.orig_shape), 'counts': mask_bytes})).bool()
    # Delete variables that are no longer needed
    del img
    del anns
    del class_ids_from_oneformer_coco
    del class_ids_from_oneformer_ade20k