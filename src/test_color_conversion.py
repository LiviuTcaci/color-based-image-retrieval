from image_processing import load_image, convert_to_hsv, convert_to_rgb, display_rgb_hsv_comparison
import os
import numpy as np

def test_color_conversion():
    # Directorul cu imagini de test
    test_dir = "tests/test_images"
    
    # Testează fiecare imagine din director
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nTestare conversie pentru: {image_file}")
            
            try:
                # Încarcă imaginea
                rgb_image = load_image(image_path)
                
                # Convertește în HSV
                hsv_image = convert_to_hsv(rgb_image)
                
                # Convertește înapoi în RGB
                rgb_converted = convert_to_rgb(hsv_image)
                
                # Verifică dacă conversia este corectă
                mse = np.mean((rgb_image - rgb_converted) ** 2)
                print(f"Mean Squared Error: {mse:.2f}")
                
                # Afișează comparația
                display_rgb_hsv_comparison(rgb_image, hsv_image, 
                                         f"Conversie RGB <-> HSV: {image_file}")
                
            except Exception as e:
                print(f"✗ Eroare: {str(e)}")

if __name__ == "__main__":
    test_color_conversion() 