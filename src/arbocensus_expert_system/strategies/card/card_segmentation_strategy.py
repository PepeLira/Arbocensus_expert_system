import numpy as np
import pdb

class CardSegmentationStrategy:
    def __init__(self, mask_extractor, object_detector):
        self.mask_extractor = mask_extractor
        self.object_detector = object_detector
        self.card_prompt = "credit card . business card ."

    def segment(self, card_image):
        if card_image.manual_mark is None:
            score, bbox = self.object_detector.classify_object(
                card_image, 
                self.card_prompt,
                target_words=["card"])
            if score is None or bbox is None:
                return card_image
            center = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]])
        else:
            x_center, y_center = card_image.manual_mark
            x_center = x_center * card_image.array.shape[1]
            y_center = y_center * card_image.array.shape[0]
            center = np.array([[x_center, y_center]])
            pad = card_image.array.shape[1] * 0.1
            bbox = np.array([x_center - pad, y_center - pad, x_center + pad, y_center + pad])
            score = 1
        
        card_mask, m_score = self.mask_extractor.extract_mask(
            card_image, 
            center, 
            np.array([1]), 
            xyxy_bbox=bbox)
        
        card_image.define_card_mask(card_mask)
        card_image.assign_classification_score(score)
        return card_image