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
        self.dap = None
        self.height = None
        self.principal_branches_height = None
        self.trunk_xyxy = None
        self.mask_xyxy = None
        self.species = None
        self.species_confidence = None

    def get_image_xyxy_bbox(self, x_padding=30):
        size = self.array.shape
        xyxy_bbox = np.array([0 + x_padding, 0, size[1] - x_padding, size[0]])
        return xyxy_bbox
    
    def get_mask(self):
        return self.tree_mask()
    
    def display_image(self, with_tree_mask=False, show=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.array)
        ax.axis('off')
        if with_tree_mask and self.tree_mask is not None:
            if self.tree_mask.crown_mask is not None and self.tree_mask.trunk_mask is not None:
                ax.imshow(show_mask(self.tree_mask.crown_mask, color_i=1))
                ax.imshow(show_mask(self.tree_mask.trunk_mask, color_i=2))
            else:
                ax.imshow(show_mask(self.tree_mask.binary_mask, color_i=1))
            if self.metrics_available():
                ax.set_title(("Classification score: {:.2f}, "
                            "DAP: {:.2f} m, Height: {:.2f} m, "
                            "Principal branches height: {:.2f} m").format(
                    self.classification_score, self.dap,
                    self.height, self.principal_branches_height))
        if show:
            plt.show()
        return fig, ax

    def save_image(self, path):
        fig, ax = self.display_image(with_tree_mask=True, show=False)
        path = path + '/masked-' + self.file
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free memory
    
    def assign_hypothesis_score(self, score):
        self.hypothesis_score = score

    def assign_classification_score(self, score):
        self.classification_score = score

    def get_search_card_image(self):
        return self.tree_mask.get_search_card_image(self.array)

    def define_tree_mask(self, mask: TreeMask):
        self.tree_mask = mask
    
    def define_dap(self, dap):
        self.dap = dap

    def define_height(self, height):
        self.height = height

    def define_principal_branches_height(self, height):
        self.principal_branches_height = height

    def define_species(self, species, confidence):
        self.species = species
        self.species_confidence = confidence

    def get_trunk_xyxy_bbox(self):
        if self.tree_mask is not None and self.tree_mask.trunk_mask is not None:
            self.trunk_xyxy = self.tree_mask.get_trunk_xyxy().tolist()
        return self.trunk_xyxy
    
    def get_mask_xyxy_bbox(self):
        if self.tree_mask is not None:
            self.mask_xyxy = self.tree_mask.get_tree_mask_xyxy().tolist()
        return self.mask_xyxy

    def metrics_available(self):
        return self.dap is not None and self.height is not None and self.principal_branches_height is not None
    
    def __str__(self):
        return f"Image {self.count} from file {self.file}"
    
    def __call__(self):
        return self.array