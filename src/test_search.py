from image_processing import search_similar_images, compute_1d_histogram, euclidean_distance, compute_2d_histogram, compute_3d_histogram_rgb, compute_2d_histogram_for_comparison
import os
import numpy as np

def rgb_1d_histogram_concat(image):
    return np.concatenate([
        compute_1d_histogram(image, 0),
        compute_1d_histogram(image, 1),
        compute_1d_histogram(image, 2)
    ])

def rgb_2d_histogram_rg(image):
    return compute_2d_histogram_for_comparison(image, 0, 1, bins=32).flatten()

def rgb_2d_histogram_rb(image):
    return compute_2d_histogram_for_comparison(image, 0, 2, bins=32).flatten()

def rgb_2d_histogram_gb(image):
    return compute_2d_histogram_for_comparison(image, 1, 2, bins=32).flatten()

def rgb_3d_histogram(image):
    return compute_3d_histogram_rgb(image, bins=8).flatten()

def test_search():
    test_dir = "tests/test_images"
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Nu există imagini de test!")
        return
    query_image = os.path.join(test_dir, "coffee.png")
    print(f"Imagine query: {os.path.basename(query_image)}\n")
    for name, hist_func in [
        ("1D RGB", rgb_1d_histogram_concat),
        ("2D RGB R-G", rgb_2d_histogram_rg),
        ("2D RGB R-B", rgb_2d_histogram_rb),
        ("2D RGB G-B", rgb_2d_histogram_gb),
        ("3D RGB", rgb_3d_histogram)
    ]:
        results = search_similar_images(
            query_image_path=query_image,
            image_paths=image_files,
            histogram_func=hist_func,
            distance_func=euclidean_distance,
            top_n=3,
            normalize=True
        )
        print(f"Top 3 imagini similare ({name}, normalizat):")
        for path, dist in results:
            print(f"{os.path.basename(path):25s}  distanță: {dist:.5f}")
        print()

if __name__ == "__main__":
    test_search() 