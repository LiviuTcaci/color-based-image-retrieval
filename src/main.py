import os
import sys
import cv2
import numpy as np
from image_processing.color_spaces import (
    load_image, save_image, rgb_to_hsv, hsv_to_rgb, verify_conversion
)
from image_processing.histogram import compute_2d_rgb_histogram, visualize_2d_histogram

def main():
    """
    Funcția principală care demonstrează funcționalitățile de bază ale procesării imaginilor.
    """
    # Verificăm dacă există directorul pentru imagini de test
    test_images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests', 'test_images')
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
        print(f"Am creat directorul pentru imagini de test: {test_images_dir}")
        print("Vă rugăm să adăugați câteva imagini în acest director pentru testare.")
        return

    # Verificăm dacă există imagini în directorul de test
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    if not test_images:
        print("Nu s-au găsit imagini în directorul de test.")
        print("Vă rugăm să adăugați câteva imagini în directorul:", test_images_dir)
        return

    # Procesăm prima imagine găsită
    image_path = os.path.join(test_images_dir, test_images[0])
    print(f"Procesăm imaginea: {image_path}")

    try:
        # Încărcăm imaginea
        image = load_image(image_path)
        print(f"Imagine încărcată cu succes. Dimensiuni: {image.shape}")

        # Calculăm histogramele 2D RGB
        print("\nCalculăm histogramele 2D RGB...")
        hist_rg, hist_rb, hist_gb = compute_2d_rgb_histogram(image)
        print(f"Histograme calculate cu succes. Dimensiuni: {hist_rg.shape}")

        # Vizualizăm histogramele
        print("Vizualizăm histogramele...")
        vis_rg = visualize_2d_histogram(hist_rg, "RG Histogram")
        vis_rb = visualize_2d_histogram(hist_rb, "RB Histogram")
        vis_gb = visualize_2d_histogram(hist_gb, "GB Histogram")

        # Salvăm histogramele vizualizate
        base_name = os.path.splitext(test_images[0])[0]
        save_image(vis_rg, os.path.join(test_images_dir, f"{base_name}_hist_rg.jpg"))
        save_image(vis_rb, os.path.join(test_images_dir, f"{base_name}_hist_rb.jpg"))
        save_image(vis_gb, os.path.join(test_images_dir, f"{base_name}_hist_gb.jpg"))
        print("Histogramele au fost salvate cu succes")

        # Afișăm informații despre histograme
        print("\nInformații despre histograme:")
        print(f"RG Histogram - Valoare maximă: {np.max(hist_rg):.4f}")
        print(f"RB Histogram - Valoare maximă: {np.max(hist_rb):.4f}")
        print(f"GB Histogram - Valoare maximă: {np.max(hist_gb):.4f}")

    except Exception as e:
        print(f"Eroare în timpul procesării: {str(e)}")
        return

if __name__ == "__main__":
    main()
