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
from image_processing.comparison import compare_images
from image_processing.search import ImageDatabase

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

    # Inițializăm baza de date de imagini
    print("\nInițializăm baza de date de imagini...")
    db = ImageDatabase(test_images_dir)

    # Alegem imaginea de interogare
    query_idx = 0  # implicit prima imagine
    if len(sys.argv) > 1:
        try:
            query_idx = int(sys.argv[1])
            if query_idx < 0 or query_idx >= len(test_images):
                print(f"Index invalid pentru imaginea de interogare. Folosim implicit prima imagine (index 0)")
                query_idx = 0
        except ValueError:
            # Dacă argumentul este un nume de fișier, căutăm imaginea cu numele respectiv
            image_name = sys.argv[1]
            if image_name in test_images:
                query_idx = test_images.index(image_name)
            else:
                print(f"Nu s-a găsit imaginea {image_name}. Folosim implicit prima imagine (index 0)")

    # Alegem metoda de căutare
    method = 'HSV'  # implicit HSV
    if len(sys.argv) > 2:
        method = sys.argv[2]
        if method not in ['2D_RGB', '3D_RGB', 'HSV']:
            print(f"Metodă invalidă. Folosim implicit HSV")
            method = 'HSV'

    # Alegem numărul de rezultate
    top_k = 5  # implicit 5 rezultate
    if len(sys.argv) > 3:
        try:
            top_k = int(sys.argv[3])
            if top_k < 1:
                print(f"Număr invalid de rezultate. Folosim implicit 5")
                top_k = 5
        except ValueError:
            print(f"Număr invalid de rezultate. Folosim implicit 5")

    # Procesăm imaginea de interogare
    query_image_path = os.path.join(test_images_dir, test_images[query_idx])
    print(f"\nImagine de interogare: {query_image_path}")
    print(f"Metodă de căutare: {method}")
    print(f"Număr de rezultate: {top_k}")

    try:
        # Încărcăm imaginea de interogare
        query_image = load_image(query_image_path)
        print(f"Imagine încărcată cu succes. Dimensiuni: {query_image.shape}")

        # Căutăm imagini similare
        print("\nCăutăm imagini similare...")
        results = db.search_similar(query_image, method=method, top_k=top_k)

        # Afișăm rezultatele
        print("\nRezultate căutare:")
        print("----------------")
        for idx, (image_name, distance) in enumerate(results, 1):
            print(f"{idx}. {image_name} (distanță: {distance:.4f})")

        # Salvăm rezultatele într-un fișier text
        results_file = os.path.join(test_images_dir, f"search_results_{method}.txt")
        with open(results_file, 'w') as f:
            f.write(f"Rezultate căutare pentru {test_images[query_idx]}\n")
            f.write(f"Metodă: {method}\n")
            f.write("----------------\n")
            for idx, (image_name, distance) in enumerate(results, 1):
                f.write(f"{idx}. {image_name} (distanță: {distance:.4f})\n")
        print(f"\nRezultatele au fost salvate în {results_file}")

    except Exception as e:
        print(f"Eroare în timpul procesării: {str(e)}")
        return

if __name__ == "__main__":
    main()
