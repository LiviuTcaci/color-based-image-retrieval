import cv2
import numpy as np

def rgb_to_hsv(image):
    """
    Convertește o imagine din spațiul de culoare RGB în HSV.
    
    Args:
        image (numpy.ndarray): Imaginea în format RGB (BGR pentru OpenCV)
        
    Returns:
        numpy.ndarray: Imaginea convertită în spațiul HSV
    """
    # OpenCV folosește BGR în loc de RGB, deci nu trebuie să facem conversie
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hsv_to_rgb(image):
    """
    Convertește o imagine din spațiul de culoare HSV în RGB.
    
    Args:
        image (numpy.ndarray): Imaginea în format HSV
        
    Returns:
        numpy.ndarray: Imaginea convertită în spațiul RGB (BGR pentru OpenCV)
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def load_image(image_path):
    """
    Încarcă o imagine și o convertește în formatul corect.
    
    Args:
        image_path (str): Calea către imagine
        
    Returns:
        numpy.ndarray: Imaginea încărcată în format BGR
    """
    # Citim imaginea folosind OpenCV (returnează în format BGR)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nu s-a putut încărca imaginea de la calea: {image_path}")
    
    # Verificăm dacă imaginea este în format TIFF și are 4 canale (RGBA)
    if image_path.lower().endswith(('.tif', '.tiff')) and image.shape[2] == 4:
        # Convertim din RGBA în BGR eliminând canalul alpha
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    return image

def save_image(image, save_path):
    """
    Salvează o imagine la calea specificată.
    
    Args:
        image (numpy.ndarray): Imaginea de salvat (în format BGR)
        save_path (str): Calea unde să se salveze imaginea
    """
    cv2.imwrite(save_path, image)

def normalize_image(image):
    """
    Normalizează valorile pixelilor în intervalul [0, 1].
    
    Args:
        image (numpy.ndarray): Imaginea de normalizat
        
    Returns:
        numpy.ndarray: Imaginea normalizată
    """
    return image.astype(np.float32) / 255.0

def denormalize_image(image):
    """
    Convertește valorile pixelilor din intervalul [0, 1] în [0, 255].
    
    Args:
        image (numpy.ndarray): Imaginea de denormalizat
        
    Returns:
        numpy.ndarray: Imaginea denormalizată
    """
    return (image * 255).astype(np.uint8)

def verify_conversion(original_image, converted_image, tolerance=1):
    """
    Verifică dacă conversia a păstrat informația imaginii corect.
    
    Args:
        original_image (numpy.ndarray): Imaginea originală
        converted_image (numpy.ndarray): Imaginea după conversie
        tolerance (int): Toleranța pentru diferențe (în pixeli)
        
    Returns:
        bool: True dacă conversia a păstrat informația corect, False altfel
    """
    # Verificăm dacă dimensiunile sunt identice
    if original_image.shape != converted_image.shape:
        return False
    
    # Calculăm diferența absolută între imagini
    diff = cv2.absdiff(original_image, converted_image)
    
    # Verificăm dacă există diferențe mai mari decât toleranța
    return np.all(diff <= tolerance)
