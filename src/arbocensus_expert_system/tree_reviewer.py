from arbocensus_expert_system.strategies.tree.tree_segmentation_strategy import TreeSegmentationStrategy
from arbocensus_expert_system.strategies.card.card_segmentation_strategy import CardSegmentationStrategy
from arbocensus_expert_system.models.card_image import CardImage
from arbocensus_expert_system.models.tree_metrics import TreeMetrics
from arbocensus_expert_system.helpers.validation_helper import ValidationHelper as vh
import os
from arbocensus_expert_system.config import get_env

class TreeReviewer:
    def __init__(self, tree_segmentation_strategy: TreeSegmentationStrategy, card_segmentation_strategy: CardSegmentationStrategy):
        self.tree_segmentation_strategy = tree_segmentation_strategy
        self.card_segmentation_strategy = card_segmentation_strategy
        self.error_tags = None

    def review_tree(self, tree_image, plot=False):  
        self.error_tags = []
        
        tree_image = self.segment_tree_image(tree_image, plot=plot)
        search_card_image, card_marks = tree_image.get_search_card_image()
        if card_marks is not None:
            tree_image.update_card_mark(card_marks)
        segmented_card = self.review_card(search_card_image, tree_image)

        tree_metrics = TreeMetrics(tree_image)
        tree_metrics.assign_error_tags(self.error_tags)

        if "No card detected" not in self.error_tags:
            mm_per_pixel = segmented_card.mm_per_pixel
            tree_metrics = self.measure_tree_metrics(tree_metrics, mm_per_pixel)
        
        if plot:
            segmented_card.display_image(with_mask=True)
            tree_metrics.tree_image.display_image(with_tree_mask=True)

        tree_metrics.group_metrics()

        return tree_metrics

    def review_card(self, image, tree_image):
        card_image = CardImage(image, mark=tree_image.card_mark) 
        if len(card_image.array) > 5:
            card_image = self.card_segmentation_strategy.segment(card_image)
        pixel_diameter = tree_image.tree_mask.get_lower_diameter()
        validation_result = vh.validate_segmented_card(card_image, pixel_diameter)
        self.error_tags += validation_result
        return card_image
    
    def segment_tree_image(self, tree_image, plot=False):
        if plot:
            tree_image.display_image(with_tree_mask=False)
        tree_image = self.tree_segmentation_strategy.segment(tree_image, plot=plot)

        validation_result = vh.validate_tree_image(tree_image)
        self.error_tags += validation_result
        return tree_image
    
    def measure_tree_metrics(self, tree_metrics, mm_per_pixel):
        tree_metrics.define_metrics(mm_per_pixel)

        validation_result = vh.validate_metrics(tree_metrics)
        tree_metrics.assign_error_tags(validation_result)

        return tree_metrics

    def save_tree_figure(self, tree_image, output_path):
        if not os.path.exists(os.path.join(output_path, 'mask_results')):
            os.makedirs(os.path.join(output_path, 'mask_results'))
    
        mask_results_path = os.path.join(output_path, 'mask_results')
        tree_image.save_image(mask_results_path)

    def get_error_tags(self):
        return self.error_tags

# Example of usage
if __name__ == "__main__":
    from image_loader import ImageLoader
    from models.mask_extractor_sam import MaskExtractorSAM
    from models.monocular_depth_dam import MonocularDepthDAM
    from models.gdino_object_detector import GDinoObjectDetector
    from strategies.tree.first_tree_segmenter import FirstTreeSegmenter

    tree_segmentation_strategy = FirstTreeSegmenter(MaskExtractorSAM(), 
                                                        MonocularDepthDAM(), 
                                                        GDinoObjectDetector())
    card_segmentation_strategy = CardSegmentationStrategy(MaskExtractorSAM(), 
                                                        GDinoObjectDetector())

    folder_path = get_env('TEST_TREE_IMAGES_PATH')
    image_loader = ImageLoader(folder_path)
    tree_reviewer = TreeReviewer(tree_segmentation_strategy, card_segmentation_strategy)

    count = 0

    image = next(image_loader.load_image(image_file='84436-0.jpg'))
    tree_validation_result = tree_reviewer.review_tree(image, plot=True)