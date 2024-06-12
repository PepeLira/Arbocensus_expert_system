import numpy as np
import matplotlib.pyplot as plt
from .tree_mask import TreeMask
from .helpers.model_helpers import show_mask

class TreeImage:
    def __init__(self, image_array, file, count, pil_image):
        self.array = image_array
        self.file = file
        self.count = count
        self.pil_image = pil_image
        self.shape = image_array.shape
        self.tree_mask = None
        self.classification_score = None
        self.hypothesis_score = None

    def get_image_xyxy_bbox(self):
        size = self.array.shape
        xyxy_bbox = np.array([0, 0, size[1], size[0]])
        return xyxy_bbox
    
    def get_mask(self):
        return self.tree_mask()
    
    def display_image(self, with_tree_mask=False):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.array)
        if with_tree_mask and self.tree_mask is not None:
            if self.tree_mask.crown_mask is not None and self.tree_mask.trunk_mask is not None:
                show_mask(self.tree_mask.crown_mask, plt.gca(), color_i=1) 
                show_mask(self.tree_mask.trunk_mask, plt.gca(), color_i=2)
            else:
                show_mask(self.tree_mask.binary_mask, plt.gca(), color_i=1)
        plt.show()

    def define_tree_mask(self, mask: TreeMask):
        self.tree_mask = mask
    
    def assign_hypothesis_score(self, score):
        self.hypothesis_score = score

    def assign_classification_score(self, score):
        self.classification_score = score

    def get_search_card_image(self):
        return self.tree_mask.get_search_card_image(self.array)
        

    def __str__(self):
        return f"Image {self.count} from file {self.file}"
    
    def __call__(self):
        return self.array