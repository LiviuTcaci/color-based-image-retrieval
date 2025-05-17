import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Loads an image and converts it to RGB format.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        numpy.ndarray: Image in RGB format
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the image cannot be loaded
    """
    # Check if file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"File does not exist: {image_path}")
    
    # Check file extension
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise ValueError(f"Unsupported file format: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Check image dimensions
    if img.size == 0:
        raise ValueError(f"Image is empty: {image_path}")
    
    # Convert to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return rgb_img

def convert_to_hsv(rgb_image):
    """
    Converts an image from RGB to HSV color space.
    
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        
    Returns:
        numpy.ndarray: Image in HSV format
        
    Raises:
        ValueError: If the image is not in RGB format
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Image must be in RGB format (3 channels)")
    
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

def get_image_info(image):
    """
    Returns information about the image.
    
    Args:
        image (numpy.ndarray): Image in RGB or HSV format
        
    Returns:
        dict: Dictionary with image information
    """
    return {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': np.min(image),
        'max_value': np.max(image),
        'mean_value': np.mean(image),
        'channels': image.shape[2] if len(image.shape) == 3 else 1
    }

def convert_to_rgb(hsv_image):
    """
    Converts an image from HSV to RGB color space.
    
    Args:
        hsv_image (numpy.ndarray): Image in HSV format
        
    Returns:
        numpy.ndarray: Image in RGB format
        
    Raises:
        ValueError: If the image is not in HSV format
    """
    if len(hsv_image.shape) != 3 or hsv_image.shape[2] != 3:
        raise ValueError("Image must be in HSV format (3 channels)")
    
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def display_rgb_hsv_comparison(rgb_image, hsv_image, title="RGB vs HSV Comparison"):
    """
    Displays the original RGB image, its HSV version, and the conversion back to RGB.
    
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        hsv_image (numpy.ndarray): Image in HSV format
        title (str): Figure title
    """
    plt.figure(figsize=(18, 6))
    
    # Display original RGB image
    plt.subplot(131)
    plt.imshow(rgb_image)
    plt.title('Original RGB')
    plt.axis('off')
    
    # Display HSV image
    plt.subplot(132)
    # Convert HSV back to RGB for display
    hsv_display = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    plt.imshow(hsv_display)
    plt.title('HSV')
    plt.axis('off')
    
    # Display conversion back to RGB
    plt.subplot(133)
    rgb_converted = convert_to_rgb(hsv_image)
    plt.imshow(rgb_converted)
    plt.title('RGB (converted from HSV)')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def compute_2d_histogram(image, channel1, channel2, bins=256):
    """
    Computes 2D histogram for two color channels.
    
    Args:
        image (numpy.ndarray): Image in RGB format
        channel1 (int): Index of first channel (0=R, 1=G, 2=B)
        channel2 (int): Index of second channel (0=R, 1=G, 2=B)
        bins (int): Number of bins for histogram
        
    Returns:
        numpy.ndarray: Normalized 2D histogram
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be in RGB format (3 channels)")
    
    # Extract channels
    ch1 = image[:, :, channel1]
    ch2 = image[:, :, channel2]
    
    # Compute 2D histogram
    hist = np.zeros((bins, bins))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[ch1[i,j], ch2[i,j]] += 1
    
    # Normalize histogram using log scale for better visualization
    hist = np.log1p(hist)  # log1p(x) = log(1 + x) to avoid log(0)
    hist = hist / np.max(hist)  # normalize to [0,1]
    
    return hist

def display_2d_rgb_histograms(rgb_image, title="2D RGB Histograms"):
    """
    Displays 2D histograms for all RGB planes.
    
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        title (str): Figure title
    """
    # Channel names
    channels = ['R', 'G', 'B']
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Planes to analyze
    planes = [(0,1), (0,2), (1,2)]  # R-G, R-B, G-B
    plane_names = ['R-G', 'R-B', 'G-B']
    
    for idx, ((ch1, ch2), plane_name) in enumerate(zip(planes, plane_names)):
        # Compute 2D histogram
        hist = compute_2d_histogram(rgb_image, ch1, ch2)
        
        # Display histogram
        plt.subplot(1, 3, idx + 1)
        plt.imshow(hist, cmap='viridis', origin='lower')  # use viridis for better visualization
        plt.colorbar(label='Normalized frequency (log scale)')
        plt.xlabel(channels[ch2])
        plt.ylabel(channels[ch1])
        plt.title(f'Plane {plane_name}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_rgb_distribution(image_path):
    """
    Analyzes RGB color distribution of an image using 2D histograms.
    
    Args:
        image_path (str): Path to the image
    """
    # Load image
    rgb_image = load_image(image_path)
    
    # Display 2D histograms
    display_2d_rgb_histograms(rgb_image, f"2D RGB Histograms: {Path(image_path).name}")

def compute_1d_histogram(image, channel, bins=256):
    """
    Computes 1D histogram for a color channel.
    
    Args:
        image (numpy.ndarray): Image in RGB format
        channel (int): Channel index (0=R, 1=G, 2=B)
        bins (int): Number of bins for histogram
    Returns:
        numpy.ndarray: 1D histogram
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be in RGB format (3 channels)")
    channel_data = image[:, :, channel].flatten()
    hist, _ = np.histogram(channel_data, bins=bins, range=(0, 256))
    return hist

def display_1d_rgb_histograms(rgb_image, title="1D RGB Histograms"):
    """
    Displays 1D histograms for R, G, B channels.
    
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        title (str): Figure title
    """
    colors = ['red', 'green', 'blue']
    channel_names = ['R', 'G', 'B']
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = compute_1d_histogram(rgb_image, i)
        plt.plot(hist, color=color, label=channel_names[i])
    plt.title(title)
    plt.xlabel('Channel value (0-255)')
    plt.ylabel('Number of pixels')
    plt.legend()
    plt.tight_layout()
    plt.show()

def display_1d_hsv_histograms(rgb_image, title="1D HSV Histograms"):
    """
    Displays 1D histograms for H, S, V channels.
    
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        title (str): Figure title
    """
    # Convert image to HSV
    hsv_image = convert_to_hsv(rgb_image)
    channel_names = ['H', 'S', 'V']
    colors = ['orange', 'purple', 'gray']
    bins = [180, 256, 256]  # H: 0-179, S: 0-255, V: 0-255
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = compute_1d_histogram(hsv_image, i, bins=bins[i])
        plt.plot(hist, color=color, label=channel_names[i])
    plt.title(title)
    plt.xlabel('Channel value (H:0-179, S/V:0-255)')
    plt.ylabel('Number of pixels')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_2d_histogram_hsv(hsv_image, channel1, channel2, bins1, bins2):
    """
    Computes 2D histogram for two HSV channels.
    Args:
        hsv_image (numpy.ndarray): Image in HSV format
        channel1 (int): Index of first channel (0=H, 1=S, 2=V)
        channel2 (int): Index of second channel (0=H, 1=S, 2=V)
        bins1 (int): Number of bins for first channel
        bins2 (int): Number of bins for second channel
    Returns:
        numpy.ndarray: Normalized 2D histogram
    """
    if len(hsv_image.shape) != 3 or hsv_image.shape[2] != 3:
        raise ValueError("Image must be in HSV format (3 channels)")
    ch1 = hsv_image[:, :, channel1].flatten()
    ch2 = hsv_image[:, :, channel2].flatten()
    hist, _, _ = np.histogram2d(ch1, ch2, bins=[bins1, bins2], range=[[0, bins1], [0, bins2]])
    hist = np.log1p(hist)
    hist = hist / np.max(hist) if np.max(hist) > 0 else hist
    return hist

def display_2d_hsv_histograms(rgb_image, title="2D HSV Histograms"):
    """
    Displays 2D histograms for H-S, H-V, S-V planes.
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        title (str): Figure title
    """
    hsv_image = convert_to_hsv(rgb_image)
    planes = [(0,1), (0,2), (1,2)]  # H-S, H-V, S-V
    plane_names = ['H-S', 'H-V', 'S-V']
    bins = [(180, 256), (180, 256), (256, 256)]
    channel_labels = [('S', 'H'), ('V', 'H'), ('V', 'S')]
    plt.figure(figsize=(15, 5))
    for idx, ((ch1, ch2), plane_name) in enumerate(zip(planes, plane_names)):
        hist = compute_2d_histogram_hsv(hsv_image, ch1, ch2, bins[idx][0], bins[idx][1])
        plt.subplot(1, 3, idx + 1)
        plt.imshow(hist, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(label='Normalized frequency (log scale)')
        plt.xlabel(channel_labels[idx][0])
        plt.ylabel(channel_labels[idx][1])
        plt.title(f'Plane {plane_name}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compute_3d_histogram_rgb(rgb_image, bins=8):
    """
    Computes 3D histogram for RGB channels.
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        bins (int): Number of bins for each channel
    Returns:
        numpy.ndarray: 3D histogram of size (bins, bins, bins)
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Image must be in RGB format (3 channels)")
    pixels = rgb_image.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=(bins, bins, bins), range=[(0, 256), (0, 256), (0, 256)])
    return hist

def display_3d_histogram_rgb(rgb_image, bins=8, threshold=0.01, title="3D RGB Histogram (scatter)"):
    """
    Displays 3D RGB histogram as a 3D scatter plot.
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        bins (int): Number of bins for each channel
        threshold (float): Relative threshold for displaying points (0-1)
        title (str): Figure title
    """
    hist = compute_3d_histogram_rgb(rgb_image, bins)
    hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
    # Coordinates where histogram exceeds threshold
    x, y, z = np.where(hist_norm > threshold)
    values = hist_norm[x, y, z]
    colors = np.stack([x, y, z], axis=1) / (bins - 1)  # normalize to [0,1] for RGB
    from mpl_toolkits.mplot3d import Axes3D  # local import to avoid warnings
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, s=values * 200, alpha=0.6)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def compute_3d_histogram_hsv(rgb_image, bins=(18, 16, 16)):
    """
    Computes 3D histogram for HSV channels.
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        bins (tuple): Number of bins for each channel (H, S, V)
    Returns:
        numpy.ndarray: 3D histogram of size (bins_H, bins_S, bins_V)
    """
    hsv_image = convert_to_hsv(rgb_image)
    pixels = hsv_image.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=bins, range=[(0, 180), (0, 256), (0, 256)])
    return hist

def display_3d_histogram_hsv(rgb_image, bins=(18, 16, 16), threshold=0.01, title="3D HSV Histogram (scatter)"):
    """
    Displays 3D HSV histogram as a 3D scatter plot.
    Args:
        rgb_image (numpy.ndarray): Image in RGB format
        bins (tuple): Number of bins for each channel (H, S, V)
        threshold (float): Relative threshold for displaying points (0-1)
        title (str): Figure title
    """
    hist = compute_3d_histogram_hsv(rgb_image, bins)
    hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
    x, y, z = np.where(hist_norm > threshold)
    values = hist_norm[x, y, z]
    # Convert H, S, V coordinates to RGB colors for scatter
    hsv_colors = np.stack([x / (bins[0] - 1) * 179, y / (bins[1] - 1) * 255, z / (bins[2] - 1) * 255], axis=1).astype(np.uint8)
    hsv_colors = hsv_colors.reshape(-1, 1, 3)
    rgb_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB).reshape(-1, 3) / 255.0
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=rgb_colors, s=values * 200, alpha=0.6)
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')
    ax.set_title(title)
    plt.tight_layout()
    plt.show() 

def euclidean_distance(hist1, hist2):
    """
    Computes Euclidean distance between two histograms (vectors).
    Args:
        hist1 (np.ndarray): First histogram
        hist2 (np.ndarray): Second histogram
    Returns:
        float: Euclidean distance
    """
    h1 = hist1.flatten().astype(np.float64)
    h2 = hist2.flatten().astype(np.float64)
    if h1.shape != h2.shape:
        raise ValueError("Histograms must have the same dimensions!")
    return np.sqrt(np.sum((h1 - h2) ** 2))

def normalize_histogram(hist):
    """
    Normalizes histogram so that sum of all values is 1.
    Args:
        hist (np.ndarray): Histogram to normalize
    Returns:
        np.ndarray: Normalized histogram
    """
    total = np.sum(hist)
    if total == 0:
        return hist
    return hist / total 

def search_similar_images(query_image_path, image_paths, histogram_func, distance_func, top_n=3, normalize=True):
    """
    Returns top N images similar to query image using histograms and a distance metric.
    Args:
        query_image_path (str): Path to query image
        image_paths (list): List of paths to database images
        histogram_func (func): Function that takes an image and returns histogram (np.ndarray)
        distance_func (func): Function that takes two histograms and returns a distance
        top_n (int): Number of similar images to return
        normalize (bool): Whether to normalize histograms
    Returns:
        list: List of tuples (image_path, distance), sorted by distance
    """
    from image_processing import load_image, normalize_histogram
    # Compute histogram for query
    query_img = load_image(query_image_path)
    query_hist = histogram_func(query_img)
    if normalize:
        query_hist = normalize_histogram(query_hist)
    results = []
    for img_path in image_paths:
        img = load_image(img_path)
        hist = histogram_func(img)
        if normalize:
            hist = normalize_histogram(hist)
        dist = distance_func(query_hist, hist)
        results.append((img_path, dist))
    results.sort(key=lambda x: x[1])
    return results[:top_n]  

def compute_2d_histogram_for_comparison(image, channel1, channel2, bins=32):
    """
    Computes 2D histogram for two color channels, correct for comparison (uses np.histogram2d).
    Args:
        image (numpy.ndarray): Image in RGB format
        channel1 (int): Index of first channel (0=R, 1=G, 2=B)
        channel2 (int): Index of second channel (0=R, 1=G, 2=B)
        bins (int): Number of bins for histogram
    Returns:
        numpy.ndarray: 2D histogram (without log/max normalization)
    """
    ch1 = image[:, :, channel1].flatten()
    ch2 = image[:, :, channel2].flatten()
    hist, _, _ = np.histogram2d(ch1, ch2, bins=[bins, bins], range=[[0, 256], [0, 256]])
    return hist

def rgb_2d_all_planes_histogram_for_comparison(image, bins=32):
    """
    Computes and concatenates 2D histograms for all RGB planes (R-G, R-B, G-B) for comparison.
    Args:
        image (numpy.ndarray): Image in RGB format
        bins (int): Number of bins for histogram
    Returns:
        numpy.ndarray: Concatenated vector with all 3 2D histograms
    """
    h_rg = compute_2d_histogram_for_comparison(image, 0, 1, bins).flatten()
    h_rb = compute_2d_histogram_for_comparison(image, 0, 2, bins).flatten()
    h_gb = compute_2d_histogram_for_comparison(image, 1, 2, bins).flatten()
    return np.concatenate([h_rg, h_rb, h_gb])  