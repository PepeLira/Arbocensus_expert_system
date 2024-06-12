import numpy as np

class CardSegmentationStrategy:
    def __init__(self, mask_extractor, object_detector):
        self.mask_extractor = mask_extractor
        self.object_detector = object_detector
        self.card_prompt = "credit card . business card ."

    def segment(self, card_image):
        score, bbox = self.object_detector.classify_object(
            card_image, 
            self.card_prompt,
            target_words=["card"])
        if score is None or bbox is None:
            return None
        bbox_center = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]])
        card_mask, score = self.mask_extractor.extract_mask(
            card_image, 
            bbox_center, 
            np.array([1]), 
            xyxy_bbox=bbox)
        
        card_image.define_card_mask(card_mask)
        card_image.assign_classification_score(score)
        return card_image