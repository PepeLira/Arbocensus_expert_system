import cv2
import matplotlib.pyplot as plt
import numpy as np
from .helpers.model_helpers import show_mask

class CardImage:
    def __init__ (self, image_array):
        self.array = image_array
        self.binary_mask = None
        self.classification_score = None
        self.standard_size = (85.6, 53.98) # (width, height)

    def define_card_mask(self, mask):
        self.binary_mask = mask

    def assign_classification_score(self, score):
        self.classification_score = score

    def get_card_image(self):
        return self.array
    
    def get_mask(self):
        return self.binary_mask
    
    def find_card_corners(self):
        mask = self.binary_mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)
        
        rect = cv2.minAreaRect(contours[0])
        corners = cv2.boxPoints(rect)
        corners = np.int0(corners)

        return corners
    
    def evaluate_mm_per_pixel(self):
        s_width, s_height = self.standard_size
        card_corners = self.find_card_corners()
        standard_card_points = np.array([[0, 0], 
                                        [s_width, 0], 
                                        [s_width, s_height], 
                                        [0, s_height]])

        romboid_sides = np.linalg.norm(
            np.diff(card_corners, axis=0), axis=1)
        romboid_long_side_index = np.argmax(romboid_sides)
        romboid_short_side_index = np.argmin(romboid_sides)

        # Orient the romboid to match the standard card points
        if romboid_long_side_index == 1:
            card_corners = np.roll(card_corners, 1, axis=0)
        elif romboid_long_side_index == 2:
            card_corners = np.roll(card_corners, 2, axis=0)
        elif romboid_long_side_index == 3:
            card_corners = np.roll(card_corners, 3, axis=0)

        affine_matrix = cv2.getPerspectiveTransform(
            card_corners.astype(np.float32), 
            standard_card_points.astype(np.float32))

        self.mm_per_pixel = np.linalg.norm(affine_matrix[:, 0])

        return self.mm_per_pixel
    
    def display_image(self, with_mask=False):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.array)
        if with_mask and self.binary_mask is not None:
            show_mask(self.binary_mask, plt.gca())
        plt.show()

    
    def __call__(self):
        return self.array