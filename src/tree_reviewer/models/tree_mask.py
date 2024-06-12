import numpy as np
import cv2
from scipy.ndimage import label
from matplotlib import pyplot as plt
from .helpers.model_helpers import moving_average
from .helpers.mask_helpers import find_longest_continuous_segment, find_middle_bottom

class TreeMask:
    def __init__(self, binary_mask, score):
        if type(binary_mask) != type(np.array([])):
            raise ValueError("The mask should be a numpy array.")
        self.binary_mask = binary_mask
        self.score = score
        self.shape = binary_mask.shape
        self.vertical_inside_points = None
        self.trunk_mask = None
        self.crown_mask = None

    def __call__(self):
        return self.binary_mask
    
    def get_thin_mask(self, factor=20):
        if not isinstance(self.binary_mask, np.ndarray):
            raise ValueError("The mask should be a numpy array.")
        if self.binary_mask.dtype != np.uint8:
            self.binary_mask = (self.binary_mask > 0).astype(np.uint8)
        if factor < 1:
            raise ValueError("The factor should be a positive integer.")
        
        # Create the structuring element for erosion
        kernel_size = (2 * factor + 1, 2 * factor + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        
        thinned_mask = cv2.erode(self.binary_mask, kernel, iterations=1)
        
        return thinned_mask

    def isolate_largest_segment(self):
        labeled_mask, num_features = label(self.binary_mask)
        
        if num_features == 0:
            return np.zeros_like(self.binary_mask, dtype=bool)
        
        segment_sizes = np.bincount(labeled_mask.flat)
        segment_sizes[0] = 0
        
        largest_segment_label = segment_sizes.argmax()
        largest_segment_mask = labeled_mask == largest_segment_label
        
        self.binary_mask = largest_segment_mask

    def get_vertical_points(self, points_n=6, padding=100):

        thin_mask = self.get_thin_mask()
        y_coords = np.argwhere(thin_mask == 1)[:, 0]
        coords = np.argwhere(thin_mask == 1)
        height, length = thin_mask.shape
        mid_length = length // 2
        
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Calcular las posiciones verticales equidistantes dentro de la máscara
        vertical_positions = np.linspace(min_y + padding, max_y - padding, points_n, dtype=int)
        
        # Buscar el punto más cercano al mid_length en x para cada posición vertical
        points = []
        for y in vertical_positions:
            x_coords = coords[coords[:, 0] == y][:, 1]
            if len(x_coords) > 0:
                x = x_coords[np.argmin(np.abs(x_coords - mid_length))]
                points.append((y, x))
        
        self.vertical_inside_points = np.array(points)

        return self.vertical_inside_points

    def get_point_labels(self):
        return np.ones(len(self.vertical_inside_points))

    def get_mask_xyxy_bbox(self, w_pad=20):
        y_coords, x_coords = np.argwhere(self.binary_mask==1).T
        height, width = self.binary_mask.shape
        #returns an xyxy bbox as a numpy array
        xyxy_bbox = np.array([max(np.min(x_coords) - w_pad, 0), 0,
                              min(np.max(x_coords) + w_pad, width), height])
        return xyxy_bbox
    
    def reshape(self, h, w, c):
        return self.binary_mask.reshape(h, w, c)

    def get_mask_pixel_diameters(self):
        y_coords = np.unique(np.argwhere(self.binary_mask)[:, 0])

        diameters = []

        for y in y_coords:
            x_cords = np.argwhere(self.binary_mask[y])
            diameter = np.max(x_cords) - np.min(x_cords)
            diameters.append(diameter)

        return diameters, y_coords
    
    def isolate_trunk_crown_masks(self, plot=False):
        diameters, y_coords = self.get_mask_pixel_diameters()
        trunk_segment = find_longest_continuous_segment(moving_average(diameters, 15))
        print(trunk_segment)
        start, end = y_coords[trunk_segment[0]], y_coords[-1] + 1

        self.trunk_mask = np.zeros_like(self.binary_mask)
        self.trunk_mask[start:end, :] = self.binary_mask[start:end, :]
        self.crown_mask = self.binary_mask.copy()
        self.crown_mask[start:end, :] = 0
        if self.crown_mask.sum() == 0:
            self.crown_mask = None
        if plot:
            self.plot_tree_diameters(y_coords, diameters, trunk_segment)

    def define_search_card_bbox(self, width_padding=50, height_padding=20):
        middle_bottom, x_diameter = find_middle_bottom(self.binary_mask)
        search_card_bbox_height = (self.binary_mask.shape[0] - middle_bottom[1])*2
        search_card_bbox_length = x_diameter + width_padding

        if search_card_bbox_height > 200:
            height_padding = -height_padding
        # Definir la caja delimitadora de búsqueda con middle_bottom en el centro en h y w
        self.search_card_bbox = np.array([int(middle_bottom[0] - search_card_bbox_length//2), 
                                    int(middle_bottom[1] - search_card_bbox_height//2 - height_padding), 
                                    int(middle_bottom[0] + search_card_bbox_length//2), 
                                    int(middle_bottom[1] + search_card_bbox_height//2 -1)])
    
    def get_search_card_image(self, image):
        self.define_search_card_bbox()
        search_card_image = image[
            self.search_card_bbox[1]:self.search_card_bbox[3], 
            self.search_card_bbox[0]:self.search_card_bbox[2]]
        return search_card_image
    
    def display_mask(self, with_points=False, with_bbox=False):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.binary_mask, cmap="gray")
        if with_points:
            self.get_vertical_points()
            plt.scatter(self.vertical_inside_points[:, 1], self.vertical_inside_points[:, 0], c='r')
        if with_bbox:
            bbox = self.get_mask_xyxy_bbox()
            plt.plot([bbox[0], bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[3], bbox[3], bbox[1], bbox[1]], 'r')
        plt.show()

    def display_crown_trunk_masks(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(self.trunk_mask, cmap='gray')
        axs[0].set_title('Trunk Mask')
        axs[1].imshow(self.crown_mask, cmap='gray')
        axs[1].set_title('Crown Mask')
        plt.show()

    def plot_tree_diameters(self, tree_y_coords, pixel_diameters, trunk_segment):
        # Plot the width profile and highlight the longest segment
        plt.plot(tree_y_coords, pixel_diameters)
        plt.axvline(tree_y_coords[trunk_segment[0]], color='r', linestyle='--', label='Start of Longest Segment')
        plt.axvline(tree_y_coords[trunk_segment[1]], color='g', linestyle='--', label='End of Longest Segment')
        plt.xlabel('Height (Pixels)')
        plt.ylabel('Diameter (Pixels)')
        plt.title('Tree Diameter in Pixels with Longest Continuous Segment Highlighted')
        plt.legend()
        plt.show()