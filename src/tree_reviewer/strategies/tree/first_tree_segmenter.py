from .tree_segmentation_strategy import TreeSegmentationStrategy
from models.mask_extractor_sam import MaskExtractorSAM
from models.monocular_depth_dam import MonocularDepthDAM
from models.gdino_object_detector import GDinoObjectDetector
from models.tree_mask import TreeMask

class FirstTreeSegmenter(TreeSegmentationStrategy):
    def __init__(self, mask_extractor: MaskExtractorSAM, depth_mapper: MonocularDepthDAM, object_classifier: GDinoObjectDetector):
        self.mask_extractor = mask_extractor
        self.depth_mapper = depth_mapper
        self.object_classifier = object_classifier
        self.tree_prompt = "a tree. a branch. a trunk"

    def segment(self, tree_image, plot=False):
        first_tree_mask, score = self.mask_extractor.get_hypothesis_mask(
            tree_image)
        first_tree_mask = TreeMask(first_tree_mask, score)
        tree_image.define_tree_mask(first_tree_mask)
        tree_image.assign_hypothesis_score(first_tree_mask.score)
        if plot:
            first_tree_mask.display_mask()

        depth_mask = self.depth_mapper.isolate_tree_depth(
            tree_image, 
            first_tree_mask,
            plot=plot)
        if plot:
            depth_mask.display_mask(with_points=True, with_bbox=True)

        filtered_ref_coords = depth_mask.get_vertical_points()
        if len(filtered_ref_coords) > 0:
            filtered_ref_labels = depth_mask.get_point_labels()
            filtered_tree_mask, score = self.mask_extractor.extract_mask(
                tree_image, 
                filtered_ref_coords, 
                filtered_ref_labels, 
                xyxy_bbox=depth_mask.get_mask_xyxy_bbox())
            filtered_tree_mask = TreeMask(filtered_tree_mask, score)
            filtered_tree_mask.isolate_largest_segment()
            if plot:
                filtered_tree_mask.display_mask()
            filtered_tree_mask.divide_trunk_crown_masks(plot=plot)
            tree_image.define_tree_mask(filtered_tree_mask)

            
        else: 
            tree_image.define_tree_mask(first_tree_mask)
        classification_score, _ = self.object_classifier.classify_object(
            tree_image, 
            self.tree_prompt)
        tree_image.assign_classification_score(classification_score)
        
        
        return tree_image

    