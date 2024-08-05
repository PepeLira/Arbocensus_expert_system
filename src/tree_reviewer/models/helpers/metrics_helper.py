import numpy as np

def dap_from_tree_mask(trunk_mask, mm_per_pixel: float, trunk_h = 0):
    if trunk_h < 1.3:
        return -1

    y_coords = np.unique(np.argwhere(trunk_mask)[:, 0])
    # Get the diameter of the tree at 1.3 meters height
    pixel_height = 1300/mm_per_pixel
    y_1_3 = np.max(y_coords) - pixel_height

    x_cords = np.argwhere(trunk_mask[int(y_1_3)])
    diameter = (np.max(x_cords) - np.min(x_cords)) * mm_per_pixel

    return diameter/1000 # Convert to meters

def height_from_tree_mask(tree_mask, mm_per_pixel: float):
    height = mask_height(tree_mask, mm_per_pixel)
    return height/1000 # Convert to meters

def principal_branches_height(trunk_mask, mm_per_pixel: float):
    height = mask_height(trunk_mask, mm_per_pixel)
    return height/1000 # Convert to meters

def mask_height(mask: np.array, mm_per_pixel: float):
    y_coords = np.unique(np.argwhere(mask)[:, 0])

    pixel_height = max(y_coords) - min(y_coords)
    height = pixel_height * mm_per_pixel

    return height