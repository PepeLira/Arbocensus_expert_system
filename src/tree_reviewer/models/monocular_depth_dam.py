from transformers import pipeline
import numpy as np
from scipy.signal import find_peaks
from .helpers.dam_helpers import plot_gmm, plot_filtered_depth_images
from .helpers.dam_helpers import remove_outliers, fit_gmm, mask_depth_image
from .helpers.dam_helpers import separate_components, filter_depth_segments
from .helpers.dam_helpers import isolate_close_values
from .helpers.model_helpers import moving_average
from .tree_mask import TreeMask
import matplotlib.pyplot as plt

class MonocularDepthDAM:
    def __init__(self):
        self.pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
        self.depth_image_array = None

    def get_depth_image_array(self, tree_image, mask = None):
        if self.depth_image_array is None:
            self.depth_image_array = self.measure_depth(tree_image.pil_image, mask = mask)
        return self.depth_image_array
    
    def get_depth_mask(self, tree_image, mask = None):
        if self.depth_image_array is None:
            self.depth_image_array = self.measure_depth(tree_image.pil_image, mask = mask)
        mask = np.where(self.depth_image_array > 0, 1, 0)
        return TreeMask(mask, 0)

    def isolate_tree_depth(self, tree_image, mask = None, plot=False):
        if self.depth_image_array is None:
            self.depth_image_array = self.measure_depth(tree_image.pil_image, mask = mask)
        separate_components = self.get_depth_components(self.depth_image_array, plot=plot)
        filtered_depth_images = filter_depth_segments(separate_components, self.depth_image_array)

        if plot:
            plot_filtered_depth_images(filtered_depth_images, self.depth_image_array)

        depth_masks, reference_depth_mask = self.im_array_2_mask(filtered_depth_images, 
                                                                self.depth_image_array)

        scores = self.evaluate_masks_scores(depth_masks, reference_depth_mask, depth_masks[0].shape)
        best_mask_index = np.argmax(scores)
        best_mask_score = scores[best_mask_index]
        best_depth_mask = depth_masks[best_mask_index]

        return best_depth_mask

    def measure_depth(self, pil_image, mask = None):
        depth_image_array = np.array(self.pipe(pil_image)["depth"])
        if mask is not None:
            binary_mask = mask.binary_mask
            masked_depth_image_array = mask_depth_image(depth_image_array, mask)
            depth_image_array = masked_depth_image_array

        return depth_image_array
    
    def filter_depth_array(self, image_array, window_size=5):
        hist, bin_edges = np.histogram(image_array, bins=256, range=(0, 255))
        hist = remove_outliers(hist)
        hist = moving_average(hist, window_size)

        # get the count statistics of the image
        mean = int(np.sum([hist[i]*i for i in range(len(hist))])/np.sum(hist))
        std = int(np.sqrt(np.sum([hist[i]*(i-mean)**2 for i in range(len(hist))])/np.sum(hist)))
        threshold = np.mean(hist) + np.std(hist)/4

        # Find peaks with the dynamic threshold
        peaks, _ = find_peaks(hist, height=threshold)

        peaks = isolate_close_values(peaks)

        hist = hist.reshape(-1, 1) #transform hist to a 2D array (index, count)
        hist = np.hstack([np.arange(len(hist)).reshape(-1, 1), hist]) #add the index as a feature
        hist = hist.astype(np.int32) #cast to int32 for the find_peaks function

        return hist, peaks
    
    def get_depth_components(self, depth_array, plot=False):
        hist, filtered_peaks = self.filter_depth_array(depth_array)
        n_components = len(filtered_peaks) if len(filtered_peaks) > 1 else 2
        depths, counts, labels = fit_gmm(hist, n_components=n_components)
        if plot:
            plot_gmm(depths, counts, labels, n_components=n_components)

        components = separate_components(depths, labels)

        return components
    
    def im_array_2_mask(self, filtered_images, reference_image):
        depth_masks = []
        for depth_image in filtered_images:
            depth_mask = np.where(depth_image > 0, 1, 0)
            depth_mask = TreeMask(depth_mask, 0)   
            depth_masks.append(depth_mask)

        depth_masks = np.array(depth_masks)
        reference_depth_mask = np.where(reference_image > 0, 1, 0)
        return depth_masks, reference_depth_mask
        
    def evaluate_masks_scores(self, masks, reference_mask, image_shape):
        scores = []
        center_x = image_shape[1] // 2
        total_pixels = np.sum(reference_mask)
        
        for mask in masks:
            percentage_true = np.sum(mask.binary_mask) / total_pixels
            mask.isolate_largest_segment()
            true_coords = np.argwhere(mask.binary_mask)
            if len(true_coords) == 0:
                scores.append(0)
                continue
            distances = np.abs(true_coords[:, 1] - center_x)
            distances = distances / np.max(distances) # Normalized scale
            exp_distances = [np.exp(d*0.7)/len(distances) for d in distances] # exp scale
            exp_distances = exp_distances / np.max(exp_distances) # Normalized exp scale

            mean_distance = np.mean(exp_distances)

            if percentage_true < 0.1:
                mean_distance *= 1000000

            height = (np.max(true_coords[:, 0]) - np.min(true_coords[:, 0]))
            
            score = height*(mean_distance)**-1

            scores.append(score)
        
        return scores


    