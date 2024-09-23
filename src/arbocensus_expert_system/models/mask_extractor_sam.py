import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
from arbocensus_expert_system.config import get_env

class MaskExtractorSAM:
    def __init__(self):
        sam_checkpoint = get_env("SAM_CHECKPOINT")
        model_type = "vit_h"
        device = self.get_device()

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

    def extract_mask(self, object_image, coordinates, labels, xyxy_bbox=None):
        self.free_gpu_memory()

        predictor = SamPredictor(self.sam)
        predictor.set_image(object_image.array)

        if xyxy_bbox is None:
            xyxy_bbox = object_image.get_image_xyxy_bbox()

        masks, scores, logits = predictor.predict(
            point_coords=coordinates,
            point_labels=labels,
            multimask_output=False,
            box=xyxy_bbox,
        )
        
        return masks[0], scores[0]
    
    def get_hypothesis_mask(self, tree_image):
        input_points, input_label = self.coordinates_hypothesis(tree_image)
        target_mask, score = self.extract_mask(tree_image, input_points, input_label)

        return target_mask, score

    def coordinates_hypothesis(self, tree_image):
        size = tree_image.shape[:2]
        center = [size[1]//2, size[0]//2]

        x_deviation = (size[1]//2) / 1.10
        y_deviation = (size[0]//2) / 1.10
        left_bottom_corner1 = [center[0] - x_deviation + 50, center[1] + (size[0]//2)/2]
        right_bottom_corner1 = [center[0] + x_deviation - 50, center[1] + (size[0]//2)/2]
        left_bottom_corner2 = [center[0] - x_deviation - 10, center[1] + (size[0]//2)/1.1]
        right_bottom_corner2 = [center[0] + x_deviation + 10, center[1] + (size[0]//2)/1.1]
        top_left_corner = [center[0] - x_deviation + 20, center[1] - y_deviation]
        top_right_corner = [center[0] + x_deviation - 20, center[1] - y_deviation]
        bottom_center = [center[0], center[1] + size[0]//6]
        top_center = [center[0], center[1] - size[0]//6]

        reference_points = [center, bottom_center, 
                            top_center, left_bottom_corner1, 
                            right_bottom_corner1, left_bottom_corner2, 
                            right_bottom_corner2, top_left_corner, 
                            top_right_corner]
        
        input_points = np.array(reference_points)
        input_label = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])

        return input_points, input_label

    def get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return "CPU"
        
    def free_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

if __name__ == "__main__":
    mask_extractor = MaskExtractorSAM()