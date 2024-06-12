from strategies.tree.tree_segmentation_strategy import TreeSegmentationStrategy
from strategies.card.card_segmentation_strategy import CardSegmentationStrategy
from models.card_image import CardImage
from helpers.validation_helper import ValidationHelper as vh
import numpy as np

class TreeReviewer:
    def __init__(self, tree_segmentation_strategy, card_segmentation_strategy):
        self.tree_segmentation_strategy = tree_segmentation_strategy
        self.card_segmentation_strategy = card_segmentation_strategy
        self.error_tags = None

    def review_tree(self, tree_image, plot=False):  
        self.error_tags = []
        tree_image = self.prepare_tree_image(tree_image, plot=plot)
        search_card_image = tree_image.get_search_card_image()
        segmented_card = self.review_card(search_card_image, plot=plot)
        if segmented_card is not None:
            mm_per_pixel = segmented_card.evaluate_mm_per_pixel()
            print(mm_per_pixel)
        print(self.error_tags)

    def review_card(self, image, plot=False):
        card_image = CardImage(image)
        if plot:
            card_image.display_image()
        segmented_card = self.card_segmentation_strategy.segment(card_image)
        validation_result = vh.validate_segmented_card(segmented_card)
        self.error_tags += validation_result
        return segmented_card
    
    def prepare_tree_image(self, tree_image, plot=False):
        if plot:
            tree_image.display_image(with_tree_mask=False)
        tree_image = self.tree_segmentation_strategy.segment(tree_image, plot=plot)
        if plot:
            tree_image.display_image(with_tree_mask=True)

        validation_result = vh.validate_tree_image(tree_image)
        self.error_tags += validation_result
        return tree_image

    def get_error_tags(self):
        return self.error_tags

# Example of usage
if __name__ == "__main__":
    from image_loader import ImageLoader
    from models.mask_extractor_sam import MaskExtractorSAM
    from models.monocular_depth_dam import MonocularDepthDAM
    from models.gdino_object_detector import GDinoObjectDetector
    from strategies.tree.tree_segmentation_strategy import TreeSegmentationStrategy
    from strategies.card.card_segmentation_strategy import CardSegmentationStrategy

    tree_segmentation_strategy = TreeSegmentationStrategy(MaskExtractorSAM(), 
                                                        MonocularDepthDAM(), 
                                                        GDinoObjectDetector())
    card_segmentation_strategy = CardSegmentationStrategy(MaskExtractorSAM(), 
                                                        GDinoObjectDetector())

    folder_path = "C:/Users/jflir/Documents/Arbocensus/ArbocensusData/20230829_24/"
    image_loader = ImageLoader(folder_path)
    tree_reviewer = TreeReviewer(tree_segmentation_strategy, card_segmentation_strategy)

    count = 0

    image = next(image_loader.load_image(image_file='13312-0.jpg'))
    tree_validation_result = tree_reviewer.review_tree(image, plot=True)


    # card_validation_result = tree_reviewer.review_card(image)

    # error_tags = tree_reviewer.get_error_tags()
    # print(error_tags)