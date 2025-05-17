from image_processing import load_image, compute_1d_histogram, compute_2d_histogram_for_comparison, compute_3d_histogram_rgb, convert_to_hsv, normalize_histogram
import os
import numpy as np

def test_histogram_normalization():
    test_dir = "tests/test_images"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No test images found!")
        return
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        print(f"\nImage: {image_file}")
        rgb_image = load_image(image_path)
        hsv_image = convert_to_hsv(rgb_image)
        # 1D RGB
        for i, name in enumerate(['R', 'G', 'B']):
            hist = compute_1d_histogram(rgb_image, i)
            hist_norm = normalize_histogram(hist)
            print(f"1D RGB {name} histogram sum: {np.sum(hist_norm):.6f}")
        # 1D HSV
        for i, name in enumerate(['H', 'S', 'V']):
            hist = compute_1d_histogram(hsv_image, i)
            hist_norm = normalize_histogram(hist)
            print(f"1D HSV {name} histogram sum: {np.sum(hist_norm):.6f}")
        # 2D RGB (R-G)
        hist2d = compute_2d_histogram_for_comparison(rgb_image, 0, 1, bins=32)
        hist2d_norm = normalize_histogram(hist2d)
        print(f"2D RGB R-G histogram sum: {np.sum(hist2d_norm):.6f}")
        # 3D RGB
        hist3d = compute_3d_histogram_rgb(rgb_image, bins=8)
        hist3d_norm = normalize_histogram(hist3d)
        print(f"3D RGB histogram sum: {np.sum(hist3d_norm):.6f}")

if __name__ == "__main__":
    test_histogram_normalization() 