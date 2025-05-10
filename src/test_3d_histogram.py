from image_processing import load_image, display_3d_histogram_rgb, display_3d_histogram_hsv
import os

def test_3d_histogram():
    """
    Testează afișarea histogramelor 3D RGB și HSV pentru toate imaginile din directorul de test.
    """
    test_dir = "tests/test_images"
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nHistogramă 3D RGB pentru: {image_file}")
            try:
                rgb_image = load_image(image_path)
                display_3d_histogram_rgb(rgb_image, bins=8, threshold=0.01, title=f"Histogramă 3D RGB: {image_file}")
                display_3d_histogram_hsv(rgb_image, bins=(18, 16, 16), threshold=0.01, title=f"Histogramă 3D HSV: {image_file}")
            except Exception as e:
                print(f"✗ Eroare: {str(e)}")

if __name__ == "__main__":
    test_3d_histogram() 