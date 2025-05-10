import numpy as np
import cv2

def compute_2d_histogram(image, bins=8):
    """
    Calculează histograma 2D pentru un plan de culoare.
    
    Args:
        image (numpy.ndarray): Imagine în format RGB sau HSV
        bins (int): Numărul de bins pentru histogramă
        
    Returns:
        numpy.ndarray: Histograma 2D
    """
    hist = cv2.calcHist([image], [0, 1], None, [bins, bins], [0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def compute_3d_histogram(image, bins=8):
    """
    Calculează histograma 3D pentru o imagine.
    
    Args:
        image (numpy.ndarray): Imagine în format RGB sau HSV
        bins (int): Numărul de bins pentru histogramă
        
    Returns:
        numpy.ndarray: Histograma 3D
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], 
                       [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten() 