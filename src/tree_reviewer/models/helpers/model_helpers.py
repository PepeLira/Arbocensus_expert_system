import numpy as np

def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def show_mask(mask, ax, color_i):
    """Displays a binary mask on an axis with a specific color."""
    # Create a colored mask
    color_mask = np.zeros((*mask.shape, 4), dtype=np.float32)  # RGBA
    color_mask[mask == 1] = [1, 0, 0, 0.5] if color_i == 1 else [0, 0, 1, 0.5]  # Red for color_i 1, Blue for color_i 2
    
    ax.imshow(color_mask)