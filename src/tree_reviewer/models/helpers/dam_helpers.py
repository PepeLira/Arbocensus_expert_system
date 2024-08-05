import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from .model_helpers import moving_average

def plot_gmm(depths, counts, labels, n_components=2):
    plt.figure(figsize=(10, 6))
    for i in range(n_components):
        component_data = depths[labels == i]
        component_counts = counts[labels == i]
        plt.plot(component_data, component_counts, label=f'Component {i+1}')

    plt.title('Gaussian Mixture Model Applied to Depth Data')
    plt.xlabel('Depth')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def plot_filtered_depth_images(filtered_depth_images, depth_image_array):
    n_images = len(filtered_depth_images) + 1
    fig, axs = plt.subplots(1, n_images, figsize=(10 * n_images, 10))
    axs[0].imshow(depth_image_array, cmap='inferno')
    axs[0].set_title('Original Depth Image')
    for i, filtered_depth_image in enumerate(filtered_depth_images):
        axs[i+1].imshow(filtered_depth_image, cmap='inferno')
        axs[i+1].set_title(f'Component {i+1}')
        axs[i+1].axis('off')
    plt.show()

def remove_outliers(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    outliers = np.where(z_scores > threshold)[0]

    for outlier in outliers:
        data[outlier] = 0

    return data

def fit_gmm(hist, n_components=2):
    # Separar los datos de profundidad y las cuentas
    depths = hist[:, 0].reshape(-1, 1)
    counts = hist[:, 1]

    # Aplicar el modelo GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0, covariance_type='diag')
    gmm.fit(depths, counts)
    labels = gmm.predict(depths)

    return depths, counts, labels

def mask_depth_image(image, mask, mask_value=0):
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == False:
                image[i][j] = mask_value
    return image

def separate_components(depths, labels):
    components = []
    for i in np.unique(labels):
        component_data = depths[labels == i]
        components.append(component_data)

    return components

def separate_depth_segments(components, depth_image_array):
    filtered_depth_images = []
    for i in range(len(components)):
        start_roi = np.min(components[i])
        end_roi = np.max(components[i])
        filtered_depth_image = depth_image_array.copy()
        filtered_depth_image[depth_image_array >= end_roi] = 0
        filtered_depth_image[depth_image_array < start_roi] = 0
        filtered_depth_images.append(filtered_depth_image)
    return filtered_depth_images

def isolate_close_values(values):
    slopes = np.diff(values) if len(values) > 1 else [0]
    slopes_mean = int(np.mean(slopes))
    slopes_threshold = slopes_mean

    groups = []
    current_group = [values[0]]
    if len(slopes) >= 3:
        for i in range(1, len(values)):
            if slopes[i-1] <= slopes_threshold:
                current_group.append(values[i])
            else:
                groups.append(current_group)
                current_group = [values[i]]
        
        groups.append(current_group)
        groups = [int(np.mean(group)) for group in groups]
    else:
        groups = values

    return groups

def kde_segmentation(data, bandwidth=None, plot=True, threshold=0.02):
    """
    Perform 1D Kernel Density Estimation (KDE) segmentation.

    Parameters:
    data (array-like): The data to be segmented.
    bandwidth (float, optional): The bandwidth for KDE. If None, it will be automatically determined.
    plot (bool, optional): If True, plots the KDE and segmentation.
    threshold (float, optional): The percentage threshold to filter low-density values.

    Returns:
    segments (list): A list of arrays, each containing a segment of the data.
    minima (array): The values of the local minima used for segmentation.
    """
    # Perform KDE
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Evaluate KDE on a grid
    x_grid = np.linspace(data.min() - 1, data.max() + 1, 1000)
    kde_values = kde(x_grid)

    # Apply threshold to filter low-density values
    threshold_value = threshold * np.max(kde_values)
    kde_values[kde_values < threshold_value] = np.nan

    # Find local minima to identify segments
    peaks, _ = find_peaks(-kde_values)

    # Calculate distances between consecutive peaks
    peak_distances = np.diff(peaks)

    # Calculate mean peak distance
    mean_peak_distance = np.mean(peak_distances)

    peaks, _ = find_peaks(-kde_values, distance=mean_peak_distance*1.1)
    minima = x_grid[peaks]

    # Function to segment data based on thresholds
    def segment_data(data, minima):
        segments = []
        last_min = data.min() - 1
        for min_val in minima:
            segments.append(data[(data > last_min) & (data <= min_val)])
            last_min = min_val
        segments.append(data[data > last_min])
        return segments

    # Segment the data
    segments = segment_data(data, minima)

    # Plot KDE and the segmentation
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, kde_values, label='KDE')
        for min_val in minima:
            plt.axvline(min_val, color='r', linestyle='--')
        plt.hist(data, bins=50, density=True, alpha=0.5, label='Data histogram')
        plt.legend()
        plt.title('1D Kernel Density Estimation Segmentation')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()

    return segments