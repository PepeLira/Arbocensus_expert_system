import numpy as np
import matplotlib.pyplot as plt
import pdb

def find_max_diameter_jump(data_array, start_padding=0.4, window_range=10):
    """
    Encuentra el salto de diámetro más grande en un array de datos.

    :param data_array: Array de diametros.
    :param start_padding: Porcentaje de padding al inicio del array.
    :param window_range: Rango de la ventana de búsqueda.
    :param tolerance: Tolerancia para la detección del salto.
    :return: Índices del salto de diámetro más grande.
    """
    n_args = len(data_array)
    start_idx = int(n_args*start_padding)
    longest_segment = (0, 0)

    deltas = np.zeros(n_args)
    for i in range(start_idx, int(n_args*0.9)):
        deltas[i] = np.abs(data_array[i] - data_array[i - window_range])

    longest_segment = (np.argmax(deltas), n_args-1)
    
    return longest_segment


def find_longest_continuous_segment(data_array, y_start, start_padding=1, window_range=10, tolerance=100):
    """
    Find the longest continuous segment where the diameter is relatively constant.

    :param data_array: List or array of width values.
    :param tolerance: Tolerance for changes in diameter (default is 5 pixels).
    :return: Start and end indices of the longest continuous segment.
    """
    n_args = len(data_array)
    deltas = np.zeros(n_args)
    for i in range(y_start, int(n_args*0.9)):
        deltas[i] = np.abs(data_array[i] - data_array[i - window_range])
    std = np.std(deltas)
    mean = np.mean(deltas)
    threshold = std

    # get the index of the first element that is greater than the mean + 2*std
    for i in range(int(n_args*0.9), y_start, -1):
        if deltas[i] > threshold:
            start = i
            break
    end = n_args - 1
    longest_segment = start, end

    return longest_segment

def find_longest_continuous_segment2(data_array, y_start, start_padding=1, window_range=4, tolerance=100):
    """
    Find the longest continuous segment where the diameter is relatively constant.

    :param data_array: List or array of width values.
    :param tolerance: Tolerance for changes in diameter (default is 5 pixels).
    :return: Start and end indices of the longest continuous segment.
    """
    max_length = 0
    current_length = 0
    n_args = len(data_array)
    start_idx = int(y_start*start_padding)
    longest_segment = (0, 0)

    for i in range(start_idx, int(n_args*0.95)):
        if abs(data_array[i] - data_array[i - window_range]) <= tolerance:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                longest_segment = (i, n_args-1)
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
    if len(x_coords) == 0:
        x_coords = coords[coords[:, 0] == np.max(coords[:, 0])][:, 1]
    x_bottom_point = np.mean(x_coords)
    x_diameter = np.max(x_coords) - np.min(x_coords)

    return np.array([x_bottom_point, y_min_bottom]), x_diameter

def get_mask_norm_xyxy(mask):
        y_coords, x_coords = np.argwhere(mask==1).T
        height, width = mask.shape
        xyxy_bbox = np.array([np.min(x_coords)/width, 
                                np.min(y_coords)/height, 
                                np.max(x_coords)/width, 
                                np.max(y_coords)/height])
        
        return xyxy_bbox