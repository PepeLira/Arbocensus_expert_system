import numpy as np
import pdb
def moving_average(data, window_size=3):
    """
    Calculates the moving average of an array of data. 

    Parameters:
    - data: Input data array.
    - window_size: Size of the window for the moving average.

    Returns:
    - Array of the same size with the moving average of the input data.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    # Convert data to float to handle NaNs
    data = np.array(data, dtype=float)
    
    # Pad the data with the edges mean at the beginning to maintain the same size array
    pad = window_size // 2
    padded_data = np.pad(data, (pad, pad), mode='edge')

    # Calculate the moving average using a sliding window
    cumsum = np.cumsum(np.insert(padded_data, 0, 0)) 
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    filtered_data = moving_avg[:len(data)]  # Ensure the result has the same size as input data
    filtered_data = np.nan_to_num(filtered_data, nan=0)
    return filtered_data

def show_mask(mask, color_i):
    """Displays a binary mask on an axis with a specific color."""
    # Create a colored mask
    color_mask = np.zeros((*mask.shape, 4), dtype=np.float32)  # RGBA
    color_mask[mask == 1] = [1, 0, 0, 0.5] if color_i == 1 else [0, 0, 1, 0.5]  # Red for color_i 1, Blue for color_i 2
    
    return color_mask



def median_filter(data, kernel_size=101):
    """
    Apply a median filter to a one-dimensional array.
    
    Parameters:
    data (array-like): Input array to filter.
    kernel_size (int): The size of the kernel (must be an odd number).
    
    Returns:
    np.ndarray: Filtered array.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
    
    filtered_data = np.copy(data)
    half_k = kernel_size // 2
    
    for i in range(half_k, len(data) - half_k):
        filtered_data[i] = np.median(data[i - half_k:i + half_k + 1])
    
    return filtered_data

def slope(pt1, pt2):
    """Calculate the slope of the line passing through two points."""
    if pt2[0] - pt1[0] == 0:
        return round(np.abs(pt2[1] - pt1[1]), 3) 
    return round(np.abs(pt2[1] - pt1[1] / pt2[0] - pt1[0]), 3)

def distance(pt1, pt2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def sort_rhomboid_corners(coords):
    # Sort the corners in the following order: 
    # tl(top-left), tr(top-right), bl(bottom-left), br(bottom-right) 
    # where tl - tr and bl -br are the long sides of the card
    # and tl - bl  tr-br are the short sides of the card
    
    coords = coords.tolist()
    slopes_i = []
    # identify the paralled sides 
    for i in range(4):
        for j in range(i+1, 4):
            slopes_i.append([slope(coords[i], coords[j]), i, j])

    slopes = np.array(slopes_i)[:,0]
    slopes_diff_i = []
    for i in range(6):
        for j in range(i+1, 6):
            slopes_diff = abs(slopes[i] - slopes[j])
            slopes_diff_i.append([slopes_diff, i, j])
    slopes_diff = np.array(slopes_diff_i)[:,0]

    # get the four lowest differences
    slopes_diff_i = np.array(slopes_diff_i)
    slopes_diff_i = slopes_diff_i[slopes_diff_i[:,0].argsort()]
    slopes_diff_i = slopes_diff_i[:2].astype(int)
    card_sides = [slopes_i[slopes_diff_i[0,1]][1:], slopes_i[slopes_diff_i[0,2]][1:], 
                slopes_i[slopes_diff_i[1,1]][1:], slopes_i[slopes_diff_i[1,2]][1:]]

    # add distance between the two sides
    for side in card_sides:
        side.append(distance(coords[side[0]], coords[side[1]]))

    #sort the sides by distance
    card_sides = np.array(card_sides)
    final_card_sides = card_sides[card_sides[:,2].argsort()].astype(int)
    
    #check if the is there any repeated corner
    corners = [final_card_sides[2,0], final_card_sides[2,1], 
                final_card_sides[3,0], final_card_sides[3,1]]
    if len(set(corners)) < 4:
        print('Repeated corner')

    tl, tr, bl, br = (coords[final_card_sides[2,0]], coords[final_card_sides[2,1]], 
                    coords[final_card_sides[3,0]], coords[final_card_sides[3,1]])

    return np.array([tl, tr, bl, br])