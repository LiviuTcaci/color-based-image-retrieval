import os
import sys
import cv2
import numpy as np
from image_processing.color_spaces import (
    load_image, save_image, rgb_to_hsv, hsv_to_rgb, verify_conversion
)
from image_processing.histogram import (
    compute_2d_rgb_histogram, compute_3d_rgb_histogram,
    visualize_2d_histogram, visualize_3d_histogram,
    compute_hsv_histogram, visualize_hsv_histogram
)

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

    # Afișăm lista de imagini disponibile
    print("\nImagini disponibile:")
    for idx, img in enumerate(test_images):
        print(f"{idx}: {img}")

    # Alegem imaginea de procesat
    image_idx = 0  # implicit prima imagine
    if len(sys.argv) > 1:
        try:
            image_idx = int(sys.argv[1])
            if image_idx < 0 or image_idx >= len(test_images):
                print(f"Index invalid. Folosim implicit prima imagine (index 0)")
                image_idx = 0
        except ValueError:
            # Dacă argumentul este un nume de fișier, căutăm imaginea cu numele respectiv
            image_name = sys.argv[1]
            if image_name in test_images:
                image_idx = test_images.index(image_name)
            else:
                print(f"Nu s-a găsit imaginea {image_name}. Folosim implicit prima imagine (index 0)")

    # Procesăm imaginea selectată
    image_path = os.path.join(test_images_dir, test_images[image_idx])
    print(f"\nProcesăm imaginea: {image_path}")

    try:
        # Încărcăm imaginea
        image = load_image(image_path)
        print(f"Imagine încărcată cu succes. Dimensiuni: {image.shape}")

        # Calculăm histogramele 2D RGB
        print("\nCalculăm histogramele 2D RGB...")
        hist_rg, hist_rb, hist_gb = compute_2d_rgb_histogram(image)
        print(f"Histograme 2D calculate cu succes. Dimensiuni: {hist_rg.shape}")

        # Vizualizăm histogramele 2D
        print("Vizualizăm histogramele 2D...")
        vis_rg = visualize_2d_histogram(hist_rg, "RG Histogram")
        vis_rb = visualize_2d_histogram(hist_rb, "RB Histogram")
        vis_gb = visualize_2d_histogram(hist_gb, "GB Histogram")

        # Salvăm histogramele 2D vizualizate
        base_name = os.path.splitext(test_images[image_idx])[0]
        save_image(vis_rg, os.path.join(test_images_dir, f"{base_name}_hist_rg.jpg"))
        save_image(vis_rb, os.path.join(test_images_dir, f"{base_name}_hist_rb.jpg"))
        save_image(vis_gb, os.path.join(test_images_dir, f"{base_name}_hist_gb.jpg"))
        print("Histogramele 2D au fost salvate cu succes")

        # Afișăm informații despre histogramele 2D
        print("\nInformații despre histogramele 2D:")
        print(f"RG Histogram - Valoare maximă: {np.max(hist_rg):.4f}")
        print(f"RB Histogram - Valoare maximă: {np.max(hist_rb):.4f}")
        print(f"GB Histogram - Valoare maximă: {np.max(hist_gb):.4f}")

        # Calculăm histograma 3D RGB
        print("\nCalculăm histograma 3D RGB...")
        hist_3d = compute_3d_rgb_histogram(image)
        print(f"Histogramă 3D calculată cu succes. Dimensiuni: {hist_3d.shape}")

        # Vizualizăm și salvăm histograma 3D
        print("Vizualizăm histograma 3D...")
        output_path = os.path.join(test_images_dir, f"{base_name}_hist_3d")
        vis_3d_images = visualize_3d_histogram(hist_3d, output_path)
        print(f"Histograma 3D a fost salvată cu succes în {len(vis_3d_images)} planuri")

        # Afișăm informații despre histograma 3D
        print("\nInformații despre histograma 3D:")
        print(f"Valoare maximă: {np.max(hist_3d):.4f}")
        print(f"Valoare medie: {np.mean(hist_3d):.4f}")
        print(f"Număr de bin-uri non-zero: {np.count_nonzero(hist_3d)}")

        # Calculăm histogramele HSV
        print("\nCalculăm histogramele HSV...")
        hist_h, hist_s, hist_v = compute_hsv_histogram(image)
        print("Histograme HSV calculate cu succes")

        # Vizualizăm și salvăm histogramele HSV
        print("Vizualizăm histogramele HSV...")
        vis_hsv = visualize_hsv_histogram(hist_h, hist_s, hist_v)
        save_image(vis_hsv, os.path.join(test_images_dir, f"{base_name}_hist_hsv.jpg"))
        print("Histogramele HSV au fost salvate cu succes")

        # Afișăm informații despre histogramele HSV
        print("\nInformații despre histogramele HSV:")
        print(f"Hue Histogram - Valoare maximă: {np.max(hist_h):.4f}")
        print(f"Saturation Histogram - Valoare maximă: {np.max(hist_s):.4f}")
        print(f"Value Histogram - Valoare maximă: {np.max(hist_v):.4f}")

    except Exception as e:
        print(f"Eroare în timpul procesării: {str(e)}")
        return

if __name__ == "__main__":
    main()
