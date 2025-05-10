from image_processing import analyze_rgb_distribution, display_2d_hsv_histograms
import os

def test_2d_histograms():
    """
    Testează generarea histogramelor 2D RGB și HSV pentru toate imaginile din directorul de test.
    """
    # Directorul cu imagini de test
    test_dir = "tests/test_images"
    
    # Testează fiecare imagine din director
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nAnaliză histogramă 2D pentru: {image_file}")
            
            try:
                # Analizează distribuția culorilor RGB (histograme 2D)
                analyze_rgb_distribution(image_path)
                # Analizează distribuția culorilor HSV (histograme 2D)
                rgb_image = analyze_rgb_distribution.__globals__['load_image'](image_path)
                display_2d_hsv_histograms(rgb_image, f"Histograme 2D HSV: {image_file}")
            except Exception as e:
                print(f"✗ Eroare: {str(e)}")

if __name__ == "__main__":
    test_2d_histograms() 