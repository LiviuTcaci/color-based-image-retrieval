import os
import sys
from image_processing.color_spaces import load_image, save_image, rgb_to_hsv, hsv_to_rgb

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
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
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

        # Convertim în HSV
        hsv_image = rgb_to_hsv(image)
        print("Conversie în HSV realizată cu succes")

        # Convertim înapoi în RGB
        rgb_image = hsv_to_rgb(hsv_image)
        print("Conversie înapoi în RGB realizată cu succes")

        # Salvăm imaginile procesate
        base_name = os.path.splitext(test_images[0])[0]
        save_image(hsv_image, os.path.join(test_images_dir, f"{base_name}_hsv.jpg"))
        save_image(rgb_image, os.path.join(test_images_dir, f"{base_name}_rgb.jpg"))
        print("Imaginile procesate au fost salvate cu succes")

    except Exception as e:
        print(f"Eroare în timpul procesării: {str(e)}")
        return

if __name__ == "__main__":
    main()
