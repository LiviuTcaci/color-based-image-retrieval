import matplotlib.pyplot as plt
import numpy as np

def plot_2d_histogram(hist, title="Histograma 2D"):
    """
    Vizualizează o histogramă 2D.
    
    Args:
        hist (numpy.ndarray): Histograma 2D
        title (str): Titlul plotului
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(hist.reshape(8, 8), cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_3d_histogram(hist, title="Histograma 3D"):
    """
    Vizualizează o histogramă 3D.
    
    Args:
        hist (numpy.ndarray): Histograma 3D
        title (str): Titlul plotului
    """
    plt.figure(figsize=(10, 8))
    plt.plot(hist)
    plt.title(title)
    plt.xlabel('Bin')
    plt.ylabel('Frecvență')
    plt.show() 