import numpy as np
import pdb

def find_longest_continuous_segment(data_array, start_padding=0.4, window_range=10, tolerance=5):
    """
    Find the longest continuous segment where the diameter is relatively constant.

    :param data_array: List or array of width values.
    :param tolerance: Tolerance for changes in diameter (default is 5 pixels).
    :return: Start and end indices of the longest continuous segment.
    """
    max_length = 0
    current_length = 0
    n_args = len(data_array)
    start_idx = int(n_args*start_padding)
    longest_segment = (0, 0)

    for i in range(start_idx, n_args):
        if abs(data_array[i] - data_array[i - window_range]) <= tolerance:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                longest_segment = (start_idx, i)
        else:
            current_length = 0
            start_idx = i
    
    return longest_segment

def find_middle_bottom(mask, padding=20):
    """
    Encuentra el punto medio en la parte inferior de una máscara.

    :param mask: Máscara binaria.
    :return: Coordenadas del punto medio en la parte inferior.
    """
    coords = np.argwhere(mask==1)
    y_min_bottom = np.max(coords[:, 0]) - padding
    x_coords = coords[coords[:, 0] == y_min_bottom][:, 1]
    x_bottom_point = np.mean(x_coords)
    x_diameter = np.max(x_coords) - np.min(x_coords)

    return np.array([x_bottom_point, y_min_bottom]), x_diameter