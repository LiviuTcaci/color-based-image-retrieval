from image_processing import load_image, display_1d_rgb_histograms, display_1d_hsv_histograms
import os

def test_1d_histograms():
    """
    Tests the display of 1D RGB and HSV histograms for all images in the test directory.
    """
    test_dir = "tests/test_images"
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\n1D RGB Histogram for: {image_file}")
            try:
                rgb_image = load_image(image_path)
                display_1d_rgb_histograms(rgb_image, f"1D RGB Histogram: {image_file}")
                display_1d_hsv_histograms(rgb_image, f"1D HSV Histogram: {image_file}")
            except Exception as e:
                print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    test_1d_histograms() 