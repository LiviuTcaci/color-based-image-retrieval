from image_processing import load_image, display_1d_rgb_histograms, display_1d_hsv_histograms
import os

def test_1d_histograms():
    """
    Testează afișarea histogramelor 1D RGB și HSV pentru toate imaginile din directorul de test.
    """
    test_dir = "tests/test_images"
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nHistogramă 1D RGB pentru: {image_file}")
            try:
                rgb_image = load_image(image_path)
                display_1d_rgb_histograms(rgb_image, f"Histogramă 1D RGB: {image_file}")
                display_1d_hsv_histograms(rgb_image, f"Histogramă 1D HSV: {image_file}")
            except Exception as e:
                print(f"✗ Eroare: {str(e)}")

if __name__ == "__main__":
    test_1d_histograms() 