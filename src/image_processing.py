import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Încarcă o imagine și o convertește în format RGB.
    
    Args:
        image_path (str): Calea către imagine
        
    Returns:
        numpy.ndarray: Imaginea în format RGB
        
    Raises:
        FileNotFoundError: Dacă fișierul nu există
        ValueError: Dacă imaginea nu poate fi încărcată
    """
    # Verifică dacă fișierul există
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Fișierul nu există: {image_path}")
    
    # Verifică extensia fișierului
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise ValueError(f"Format de fișier neacceptat: {image_path}")
    
    # Încarcă imaginea
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Nu s-a putut încărca imaginea: {image_path}")
    
    # Verifică dimensiunile imaginii
    if img.size == 0:
        raise ValueError(f"Imaginea este goală: {image_path}")
    
    # Convertește în RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return rgb_img

def convert_to_hsv(rgb_image):
    """
    Convertește o imagine din RGB în HSV.
    
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        
    Returns:
        numpy.ndarray: Imagine în format HSV
        
    Raises:
        ValueError: Dacă imaginea nu este în format RGB
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Imaginea trebuie să fie în format RGB (3 canale)")
    
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

def get_image_info(image):
    """
    Returnează informații despre imagine.
    
    Args:
        image (numpy.ndarray): Imagine în format RGB sau HSV
        
    Returns:
        dict: Dicționar cu informații despre imagine
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
    Convertește o imagine din HSV în RGB.
    
    Args:
        hsv_image (numpy.ndarray): Imagine în format HSV
        
    Returns:
        numpy.ndarray: Imagine în format RGB
        
    Raises:
        ValueError: Dacă imaginea nu este în format HSV
    """
    if len(hsv_image.shape) != 3 or hsv_image.shape[2] != 3:
        raise ValueError("Imaginea trebuie să fie în format HSV (3 canale)")
    
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def display_rgb_hsv_comparison(rgb_image, hsv_image, title="Comparație RGB vs HSV"):
    """
    Afișează imaginea originală în RGB, versiunea sa în HSV și conversia înapoi în RGB.
    
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        hsv_image (numpy.ndarray): Imagine în format HSV
        title (str): Titlul figurii
    """
    plt.figure(figsize=(18, 6))
    
    # Afișează imaginea RGB originală
    plt.subplot(131)
    plt.imshow(rgb_image)
    plt.title('RGB Original')
    plt.axis('off')
    
    # Afișează imaginea HSV
    plt.subplot(132)
    # Convertim HSV înapoi în RGB pentru afișare
    hsv_display = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    plt.imshow(hsv_display)
    plt.title('HSV')
    plt.axis('off')
    
    # Afișează conversia înapoi în RGB
    plt.subplot(133)
    rgb_converted = convert_to_rgb(hsv_image)
    plt.imshow(rgb_converted)
    plt.title('RGB (convertit din HSV)')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def compute_2d_histogram(image, channel1, channel2, bins=256):
    """
    Calculează histograma 2D pentru două canale de culoare.
    
    Args:
        image (numpy.ndarray): Imagine în format RGB
        channel1 (int): Indexul primului canal (0=R, 1=G, 2=B)
        channel2 (int): Indexul celui de-al doilea canal (0=R, 1=G, 2=B)
        bins (int): Numărul de bins pentru histogramă
        
    Returns:
        numpy.ndarray: Histograma 2D normalizată
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Imaginea trebuie să fie în format RGB (3 canale)")
    
    # Extrage canalele
    ch1 = image[:, :, channel1]
    ch2 = image[:, :, channel2]
    
    # Calculează histograma 2D
    hist = np.zeros((bins, bins))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[ch1[i,j], ch2[i,j]] += 1
    
    # Normalizează histograma folosind log scale pentru o mai bună vizualizare
    hist = np.log1p(hist)  # log1p(x) = log(1 + x) pentru a evita log(0)
    hist = hist / np.max(hist)  # normalizare la [0,1]
    
    return hist

def display_2d_rgb_histograms(rgb_image, title="Histograme 2D RGB"):
    """
    Afișează histogramele 2D pentru toate planurile RGB.
    
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        title (str): Titlul figurii
    """
    # Numele canalelor
    channels = ['R', 'G', 'B']
    
    # Creează figura
    plt.figure(figsize=(15, 5))
    
    # Planurile de analizat
    planes = [(0,1), (0,2), (1,2)]  # R-G, R-B, G-B
    plane_names = ['R-G', 'R-B', 'G-B']
    
    for idx, ((ch1, ch2), plane_name) in enumerate(zip(planes, plane_names)):
        # Calculează histograma 2D
        hist = compute_2d_histogram(rgb_image, ch1, ch2)
        
        # Afișează histograma
        plt.subplot(1, 3, idx + 1)
        plt.imshow(hist, cmap='viridis', origin='lower')  # folosim viridis pentru o mai bună vizualizare
        plt.colorbar(label='Frecvență normalizată (log scale)')
        plt.xlabel(channels[ch2])
        plt.ylabel(channels[ch1])
        plt.title(f'Plan {plane_name}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_rgb_distribution(image_path):
    """
    Analizează distribuția culorilor RGB ale unei imagini folosind histograme 2D.
    
    Args:
        image_path (str): Calea către imagine
    """
    # Încarcă imaginea
    rgb_image = load_image(image_path)
    
    # Afișează imaginea originală
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.title('Imagine originală')
    plt.axis('off')
    plt.show()
    
    # Afișează histogramele 2D
    display_2d_rgb_histograms(rgb_image, f"Histograme 2D RGB: {Path(image_path).name}")

def compute_1d_histogram(image, channel, bins=256):
    """
    Calculează histograma 1D pentru un canal de culoare.
    
    Args:
        image (numpy.ndarray): Imagine în format RGB
        channel (int): Indexul canalului (0=R, 1=G, 2=B)
        bins (int): Numărul de bins pentru histogramă
    Returns:
        numpy.ndarray: Histograma 1D
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Imaginea trebuie să fie în format RGB (3 canale)")
    channel_data = image[:, :, channel].flatten()
    hist, _ = np.histogram(channel_data, bins=bins, range=(0, 256))
    return hist

def display_1d_rgb_histograms(rgb_image, title="Histograme 1D RGB"):
    """
    Afișează histogramele 1D pentru canalele R, G, B.
    
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        title (str): Titlul figurii
    """
    colors = ['red', 'green', 'blue']
    channel_names = ['R', 'G', 'B']
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = compute_1d_histogram(rgb_image, i)
        plt.plot(hist, color=color, label=channel_names[i])
    plt.title(title)
    plt.xlabel('Valoare canal (0-255)')
    plt.ylabel('Număr pixeli')
    plt.legend()
    plt.tight_layout()
    plt.show()

def display_1d_hsv_histograms(rgb_image, title="Histograme 1D HSV"):
    """
    Afișează histogramele 1D pentru canalele H, S, V.
    
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        title (str): Titlul figurii
    """
    # Convertește imaginea în HSV
    hsv_image = convert_to_hsv(rgb_image)
    channel_names = ['H', 'S', 'V']
    colors = ['orange', 'purple', 'gray']
    bins = [180, 256, 256]  # H: 0-179, S: 0-255, V: 0-255
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = compute_1d_histogram(hsv_image, i, bins=bins[i])
        plt.plot(hist, color=color, label=channel_names[i])
    plt.title(title)
    plt.xlabel('Valoare canal (H:0-179, S/V:0-255)')
    plt.ylabel('Număr pixeli')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_2d_histogram_hsv(hsv_image, channel1, channel2, bins1, bins2):
    """
    Calculează histograma 2D pentru două canale din HSV.
    Args:
        hsv_image (numpy.ndarray): Imagine în format HSV
        channel1 (int): Indexul primului canal (0=H, 1=S, 2=V)
        channel2 (int): Indexul celui de-al doilea canal (0=H, 1=S, 2=V)
        bins1 (int): Numărul de bins pentru primul canal
        bins2 (int): Numărul de bins pentru al doilea canal
    Returns:
        numpy.ndarray: Histograma 2D normalizată
    """
    if len(hsv_image.shape) != 3 or hsv_image.shape[2] != 3:
        raise ValueError("Imaginea trebuie să fie în format HSV (3 canale)")
    ch1 = hsv_image[:, :, channel1].flatten()
    ch2 = hsv_image[:, :, channel2].flatten()
    hist, _, _ = np.histogram2d(ch1, ch2, bins=[bins1, bins2], range=[[0, bins1], [0, bins2]])
    hist = np.log1p(hist)
    hist = hist / np.max(hist) if np.max(hist) > 0 else hist
    return hist

def display_2d_hsv_histograms(rgb_image, title="Histograme 2D HSV"):
    """
    Afișează histogramele 2D pentru planurile H-S, H-V, S-V.
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        title (str): Titlul figurii
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
        plt.colorbar(label='Frecvență normalizată (log scale)')
        plt.xlabel(channel_labels[idx][0])
        plt.ylabel(channel_labels[idx][1])
        plt.title(f'Plan {plane_name}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compute_3d_histogram_rgb(rgb_image, bins=8):
    """
    Calculează histograma 3D pentru canalele RGB.
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        bins (int): Numărul de bins pentru fiecare canal
    Returns:
        numpy.ndarray: Histograma 3D de dimensiune (bins, bins, bins)
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Imaginea trebuie să fie în format RGB (3 canale)")
    pixels = rgb_image.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=(bins, bins, bins), range=[(0, 256), (0, 256), (0, 256)])
    return hist

def display_3d_histogram_rgb(rgb_image, bins=8, threshold=0.01, title="Histogramă 3D RGB (scatter)"):
    """
    Afișează histograma 3D RGB ca scatter plot 3D.
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        bins (int): Numărul de bins pentru fiecare canal
        threshold (float): Prag relativ pentru afișarea punctelor (0-1)
        title (str): Titlul figurii
    """
    hist = compute_3d_histogram_rgb(rgb_image, bins)
    hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
    # Coordonate unde histograma depășește pragul
    x, y, z = np.where(hist_norm > threshold)
    values = hist_norm[x, y, z]
    colors = np.stack([x, y, z], axis=1) / (bins - 1)  # normalizare la [0,1] pentru RGB
    from mpl_toolkits.mplot3d import Axes3D  # import local pentru a evita warning-uri
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
    Calculează histograma 3D pentru canalele HSV.
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        bins (tuple): Numărul de bins pentru fiecare canal (H, S, V)
    Returns:
        numpy.ndarray: Histograma 3D de dimensiune (bins_H, bins_S, bins_V)
    """
    hsv_image = convert_to_hsv(rgb_image)
    pixels = hsv_image.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=bins, range=[(0, 180), (0, 256), (0, 256)])
    return hist

def display_3d_histogram_hsv(rgb_image, bins=(18, 16, 16), threshold=0.01, title="Histogramă 3D HSV (scatter)"):
    """
    Afișează histograma 3D HSV ca scatter plot 3D.
    Args:
        rgb_image (numpy.ndarray): Imagine în format RGB
        bins (tuple): Numărul de bins pentru fiecare canal (H, S, V)
        threshold (float): Prag relativ pentru afișarea punctelor (0-1)
        title (str): Titlul figurii
    """
    hist = compute_3d_histogram_hsv(rgb_image, bins)
    hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
    x, y, z = np.where(hist_norm > threshold)
    values = hist_norm[x, y, z]
    # Convertim coordonatele H, S, V în culoare RGB pentru scatter
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
    Calculează distanța Euclidiană între două histograme (vectori).
    Args:
        hist1 (np.ndarray): Prima histogramă
        hist2 (np.ndarray): A doua histogramă
    Returns:
        float: Distanța Euclidiană
    """
    h1 = hist1.flatten().astype(np.float64)
    h2 = hist2.flatten().astype(np.float64)
    if h1.shape != h2.shape:
        raise ValueError("Histogramele trebuie să aibă aceeași dimensiune!")
    return np.sqrt(np.sum((h1 - h2) ** 2))

def normalize_histogram(hist):
    """
    Normalizează histograma astfel încât suma tuturor valorilor să fie 1.
    Args:
        hist (np.ndarray): Histograma de normalizat
    Returns:
        np.ndarray: Histograma normalizată
    """
    total = np.sum(hist)
    if total == 0:
        return hist
    return hist / total 

def search_similar_images(query_image_path, image_paths, histogram_func, distance_func, top_n=3, normalize=True):
    """
    Returnează top N imagini similare cu imaginea query, folosind histograme și o metrică de distanță.
    Args:
        query_image_path (str): Calea către imaginea query
        image_paths (list): Liste cu căi către imaginile din baza de date
        histogram_func (func): Funcție care primește o imagine și returnează histograma (np.ndarray)
        distance_func (func): Funcție care primește două histograme și returnează o distanță
        top_n (int): Numărul de imagini similare de returnat
        normalize (bool): Dacă să normalizeze histogramele
    Returns:
        list: Liste de tuple (cale_imagine, distanță), sortate crescător după distanță
    """
    from image_processing import load_image, normalize_histogram
    # Calculează histograma pentru query
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