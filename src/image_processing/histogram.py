import numpy as np
import cv2

def compute_2d_rgb_histogram(image, bins=32):
    """
    Calculează histograma 2D pentru perechi de canale RGB.
    
    Args:
        image (numpy.ndarray): Imaginea în format BGR (OpenCV)
        bins (int): Numărul de bin-uri pentru fiecare canal (implicit 32)
        
    Returns:
        tuple: (hist_rg, hist_rb, hist_gb) - histogramele pentru perechile RG, RB și GB
    """
    # Verificăm dacă imaginea este color
    if len(image.shape) != 3:
        raise ValueError("Imaginea trebuie să fie color (3 canale)")
    
    # Separăm canalele BGR
    b, g, r = cv2.split(image)
    
    # Calculăm histogramele 2D pentru fiecare pereche de canale
    # range=[0, 256] pentru că valorile pixelilor sunt în intervalul [0, 255]
    hist_rg = cv2.calcHist([r, g], [0, 1], None, [bins, bins], [0, 256, 0, 256])
    hist_rb = cv2.calcHist([r, b], [0, 1], None, [bins, bins], [0, 256, 0, 256])
    hist_gb = cv2.calcHist([g, b], [0, 1], None, [bins, bins], [0, 256, 0, 256])
    
    # Normalizăm histogramele pentru a avea suma 1
    # Acest lucru face histogramele comparabile între ele
    hist_rg = cv2.normalize(hist_rg, None, 0, 1, cv2.NORM_MINMAX)
    hist_rb = cv2.normalize(hist_rb, None, 0, 1, cv2.NORM_MINMAX)
    hist_gb = cv2.normalize(hist_gb, None, 0, 1, cv2.NORM_MINMAX)
    
    return hist_rg, hist_rb, hist_gb

def compute_3d_rgb_histogram(image, bins=8):
    """
    Calculează histograma 3D pentru toate cele trei canale RGB.
    
    Args:
        image (numpy.ndarray): Imaginea în format BGR (OpenCV)
        bins (int): Numărul de bin-uri pentru fiecare canal (implicit 8)
        
    Returns:
        numpy.ndarray: Histograma 3D (bins x bins x bins)
    """
    # Verificăm dacă imaginea este color
    if len(image.shape) != 3:
        raise ValueError("Imaginea trebuie să fie color (3 canale)")
    
    # Separăm canalele BGR
    b, g, r = cv2.split(image)
    
    # Calculăm histograma 3D pentru toate cele trei canale
    # Folosim un număr mai mic de bin-uri pentru 3D pentru a evita probleme de memorie
    hist_3d = cv2.calcHist([r, g, b], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    
    # Normalizăm histograma pentru a avea suma 1
    hist_3d = cv2.normalize(hist_3d, None, 0, 1, cv2.NORM_MINMAX)
    
    return hist_3d

def visualize_2d_histogram(hist, title="2D Histogram"):
    """
    Vizualizează o histogramă 2D folosind o hartă de culoare.
    
    Args:
        hist (numpy.ndarray): Histograma 2D de vizualizat
        title (str): Titlul pentru vizualizare
        
    Returns:
        numpy.ndarray: Imaginea cu histograma vizualizată
    """
    # Convertim histograma în format vizualizabil
    # Folosim log scale pentru a vedea mai bine distribuția
    hist_vis = np.log(hist + 1)  # +1 pentru a evita log(0)
    
    # Normalizăm pentru vizualizare
    hist_vis = cv2.normalize(hist_vis, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convertim în uint8 pentru a putea aplica o hartă de culoare
    hist_vis = hist_vis.astype(np.uint8)
    
    # Aplicăm o hartă de culoare pentru vizualizare
    hist_vis = cv2.applyColorMap(hist_vis, cv2.COLORMAP_JET)
    
    return hist_vis

def visualize_3d_histogram(hist_3d, output_path=None):
    """
    Vizualizează o histogramă 3D salvând-o ca o serie de imagini 2D.
    
    Args:
        hist_3d (numpy.ndarray): Histograma 3D de vizualizat
        output_path (str, optional): Calea unde să se salveze imaginile
        
    Returns:
        list: Lista cu imaginile vizualizate pentru fiecare plan
    """
    # Obținem dimensiunile histogramei 3D
    bins = hist_3d.shape[0]
    
    # Lista pentru a stoca imaginile vizualizate
    vis_images = []
    
    # Pentru fiecare plan din histograma 3D
    for i in range(bins):
        # Extragem planul
        plane = hist_3d[:, :, i]
        
        # Vizualizăm planul folosind funcția pentru 2D
        vis_plane = visualize_2d_histogram(plane, f"Plan {i}")
        
        # Adăugăm imaginea în listă
        vis_images.append(vis_plane)
        
        # Salvăm imaginea dacă s-a specificat o cale
        if output_path:
            cv2.imwrite(f"{output_path}_plan_{i}.jpg", vis_plane)
    
    return vis_images
