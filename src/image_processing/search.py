import os
import numpy as np
from typing import List, Dict, Tuple
import cv2
from .histogram import (
    compute_2d_rgb_histogram, compute_3d_rgb_histogram,
    compute_hsv_histogram
)
from .comparison import (
    compare_2d_histograms, compare_3d_histograms,
    compare_hsv_histograms
)

class ImageDatabase:
    """
    Clasă pentru gestionarea bazei de date de imagini și căutarea imaginilor similare.
    """
    def __init__(self, images_dir: str):
        """
        Inițializează baza de date cu imagini din directorul specificat.
        
        Args:
            images_dir: Calea către directorul cu imagini
        """
        self.images_dir = images_dir
        self.images = []
        self.histograms = {
            '2D_RGB': [],
            '3D_RGB': [],
            'HSV': []
        }
        self._load_images()
    
    def _load_images(self):
        """
        Încarcă toate imaginile din director și calculează histogramele.
        """
        # Obținem lista de fișiere imagine
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        
        print(f"Se încarcă {len(image_files)} imagini din directorul {self.images_dir}...")
        
        for img_file in image_files:
            try:
                # Încărcăm imaginea
                img_path = os.path.join(self.images_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Nu s-a putut încărca imaginea: {img_file}")
                    continue
                
                # Calculăm histogramele
                hist_2d = compute_2d_rgb_histogram(image)
                hist_3d = compute_3d_rgb_histogram(image)
                hist_hsv = compute_hsv_histogram(image)
                
                # Salvăm imaginea și histogramele
                self.images.append((img_file, image))
                self.histograms['2D_RGB'].append(hist_2d)
                self.histograms['3D_RGB'].append(hist_3d)
                self.histograms['HSV'].append(hist_hsv)
                
                print(f"Imagine încărcată: {img_file}")
                
            except Exception as e:
                print(f"Eroare la încărcarea imaginii {img_file}: {str(e)}")
    
    def search_similar(self, query_image: np.ndarray, 
                      method: str = 'HSV',
                      top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Caută imagini similare cu imaginea de interogare.
        
        Args:
            query_image: Imaginea de interogare
            method: Metoda de comparație ('2D_RGB', '3D_RGB', sau 'HSV')
            top_k: Numărul de rezultate de returnat
            
        Returns:
            List[Tuple[str, float]]: Lista de tupluri (nume_imagine, distanță)
        """
        if method not in ['2D_RGB', '3D_RGB', 'HSV']:
            raise ValueError("Metoda trebuie să fie '2D_RGB', '3D_RGB', sau 'HSV'")
        
        # Calculăm histogramele pentru imaginea de interogare
        if method == '2D_RGB':
            query_hist = compute_2d_rgb_histogram(query_image)
            distances = []
            for hist in self.histograms['2D_RGB']:
                dist_dict = compare_2d_histograms(query_hist, hist)
                # Calculăm media distanțelor pentru toate perechile
                avg_dist = sum(dist_dict.values()) / len(dist_dict)
                distances.append(avg_dist)
                
        elif method == '3D_RGB':
            query_hist = compute_3d_rgb_histogram(query_image)
            distances = [compare_3d_histograms(query_hist, hist) 
                        for hist in self.histograms['3D_RGB']]
            
        else:  # HSV
            query_hist = compute_hsv_histogram(query_image)
            distances = []
            for hist in self.histograms['HSV']:
                dist_dict = compare_hsv_histograms(query_hist, hist)
                # Calculăm media distanțelor pentru toate canalele
                avg_dist = sum(dist_dict.values()) / len(dist_dict)
                distances.append(avg_dist)
        
        # Sortăm rezultatele după distanță
        results = list(zip([img[0] for img in self.images], distances))
        results.sort(key=lambda x: x[1])  # Sortăm după distanță
        
        return results[:top_k]
    
    def get_image(self, image_name: str) -> np.ndarray:
        """
        Returnează imaginea cu numele specificat.
        
        Args:
            image_name: Numele imaginii
            
        Returns:
            np.ndarray: Imaginea
        """
        for name, image in self.images:
            if name == image_name:
                return image
        raise ValueError(f"Imaginea {image_name} nu a fost găsită în baza de date") 