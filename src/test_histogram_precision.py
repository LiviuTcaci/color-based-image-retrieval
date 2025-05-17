from image_processing import search_similar_images, compute_1d_histogram, compute_2d_histogram_for_comparison, compute_3d_histogram_rgb, euclidean_distance
import os
import numpy as np

def rgb_1d_histogram_concat(image):
    return np.concatenate([
        compute_1d_histogram(image, 0),
        compute_1d_histogram(image, 1),
        compute_1d_histogram(image, 2)
    ])

def rgb_2d_histogram_rg(image):
    # Using R-G plane with 32 bins, correct version for comparison
    return compute_2d_histogram_for_comparison(image, 0, 1, bins=32).flatten()

def rgb_3d_histogram(image):
    return compute_3d_histogram_rgb(image, bins=8).flatten()

def test_histogram_precision():
    test_dir = "tests/test_images"
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No test images found!")
        return
    query_image = os.path.join(test_dir, "coffee.png")
    print(f"Query image: {os.path.basename(query_image)}\n")
    # 1D
    results_1d = search_similar_images(
        query_image_path=query_image,
        image_paths=image_files,
        histogram_func=rgb_1d_histogram_concat,
        distance_func=euclidean_distance,
        top_n=3,
        normalize=True
    )
    print("Top 3 similar images (1D RGB, normalized):")
    for path, dist in results_1d:
        print(f"{os.path.basename(path):25s}  distance: {dist:.5f}")
    # 2D
    results_2d = search_similar_images(
        query_image_path=query_image,
        image_paths=image_files,
        histogram_func=rgb_2d_histogram_rg,
        distance_func=euclidean_distance,
        top_n=3,
        normalize=True
    )
    print("\nTop 3 similar images (2D RGB R-G, normalized):")
    for path, dist in results_2d:
        print(f"{os.path.basename(path):25s}  distance: {dist:.5f}")
    # 3D
    results_3d = search_similar_images(
        query_image_path=query_image,
        image_paths=image_files,
        histogram_func=rgb_3d_histogram,
        distance_func=euclidean_distance,
        top_n=3,
        normalize=True
    )
    print("\nTop 3 similar images (3D RGB, normalized):")
    for path, dist in results_3d:
        print(f"{os.path.basename(path):25s}  distance: {dist:.5f}")

if __name__ == "__main__":
    test_histogram_precision()
