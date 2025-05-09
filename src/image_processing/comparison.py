import numpy as np
from typing import Tuple, List, Dict
import cv2
from .histogram import compute_2d_rgb_histogram, compute_3d_rgb_histogram, compute_hsv_histogram

def euclidean_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Calculează distanța euclidiană între două histograme.
    
    Args:
        hist1: Prima histogramă
        hist2: A doua histogramă
        
    Returns:
        float: Distanța euclidiană între cele două histograme
    """
    return np.sqrt(np.sum((hist1 - hist2) ** 2))

def compare_2d_histograms(hist1: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                         hist2: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compară două seturi de histograme 2D RGB folosind distanța euclidiană.
    
    Args:
        hist1: Tuple de trei histograme 2D (RG, RB, GB) pentru prima imagine
        hist2: Tuple de trei histograme 2D (RG, RB, GB) pentru a doua imagine
        
    Returns:
        Dict[str, float]: Dicționar cu distanțele pentru fiecare pereche de histograme
    """
    distances = {}
    pairs = ['RG', 'RB', 'GB']
    
    for pair, (h1, h2) in zip(pairs, zip(hist1, hist2)):
        distances[pair] = euclidean_distance(h1, h2)
    
    return distances

def compare_3d_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compară două histograme 3D RGB folosind distanța euclidiană.
    
    Args:
        hist1: Histograma 3D RGB pentru prima imagine
        hist2: Histograma 3D RGB pentru a doua imagine
        
    Returns:
        float: Distanța euclidiană între cele două histograme 3D
    """
    return euclidean_distance(hist1, hist2)

def compare_hsv_histograms(hist1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                          hist2: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compară două seturi de histograme HSV folosind distanța euclidiană.
    
    Args:
        hist1: Tuple de trei histograme (H, S, V) pentru prima imagine
        hist2: Tuple de trei histograme (H, S, V) pentru a doua imagine
        
    Returns:
        Dict[str, float]: Dicționar cu distanțele pentru fiecare canal HSV
    """
    distances = {}
    channels = ['H', 'S', 'V']
    
    for channel, (h1, h2) in zip(channels, zip(hist1, hist2)):
        distances[channel] = euclidean_distance(h1, h2)
    
    return distances

def compare_images(image1: np.ndarray, image2: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compară două imagini folosind toate tipurile de histograme disponibile.
    
    Args:
        image1: Prima imagine
        image2: A doua imagine
        
    Returns:
        Dict[str, Dict[str, float]]: Dicționar cu toate distanțele calculate
    """
    # Calculăm histogramele pentru ambele imagini
    hist_2d_1 = compute_2d_rgb_histogram(image1)
    hist_2d_2 = compute_2d_rgb_histogram(image2)
    
    hist_3d_1 = compute_3d_rgb_histogram(image1)
    hist_3d_2 = compute_3d_rgb_histogram(image2)
    
    hist_hsv_1 = compute_hsv_histogram(image1)
    hist_hsv_2 = compute_hsv_histogram(image2)
    
    # Calculăm toate distanțele
    distances = {
        '2D_RGB': compare_2d_histograms(hist_2d_1, hist_2d_2),
        '3D_RGB': compare_3d_histograms(hist_3d_1, hist_3d_2),
        'HSV': compare_hsv_histograms(hist_hsv_1, hist_hsv_2)
    }
    
    return distances
