import cv2
import matplotlib.pyplot as plt
import numpy as np
from .helpers.model_helpers import show_mask, distance
from .helpers.card_helpers import sort_coords

class CardImage:
    def __init__ (self, image_array, mark=None):
        self.array = image_array
        self.binary_mask = None
        self.classification_score = None
        self.standard_size = (85.6, 53.98) # (width, height)
        self.card_corners = None
        self.mm_per_pixel = None
        self.l_side = None
        self.manual_mark = mark

    def define_card_mask(self, mask):
        if mask.sum() > 0:
            self.binary_mask = mask
            self.card_corners = self.find_card_corners()
            if self.card_corners is not None:
                self.evaluate_mm_per_pixel()
                self.l_side = self.card_l_side_size()

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
        #check if corners are unique
        corners = np.unique(corners, axis=0)
        if len(corners) != 4:
            return None

        corners = sort_coords(corners)
        return corners
    
    def evaluate_mm_per_pixel(self):
        s_width, s_height = self.standard_size
        standard_card_points = np.array([[0, 0], 
                                        [s_width, 0], 
                                        [0, s_height], 
                                        [s_width, s_height]])
        ratios = [s_width/distance(self.card_corners[0], self.card_corners[1]),
                s_height/distance(self.card_corners[0], self.card_corners[2]),
                s_width/distance(self.card_corners[2], self.card_corners[3]),
                s_height/distance(self.card_corners[1], self.card_corners[3])]
        
        self.mm_per_pixel = np.mean(ratios)

        return self.mm_per_pixel
    
    def display_image(self, with_mask=False):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.array)
        if with_mask and self.binary_mask is not None:
            plt.imshow(show_mask(self.binary_mask, color_i=1))
            plt.title("Classification score: {:.2f}".format(self.classification_score))
            #plot the card corners
            if self.card_corners is not None:
                corners = np.vstack([self.card_corners, self.card_corners[0]])
                plt.plot(corners[:, 0], corners[:, 1], 'bo')
        plt.show()

    def is_valid_card(self):
        return self.binary_mask is not None and self.binary_mask.sum() > 0

    def card_l_side_size(self):
        return np.linalg.norm(self.card_corners[0] - self.card_corners[1])

    
    def __call__(self):
        return self.array