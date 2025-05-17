from image_processing import load_image, compute_1d_histogram, compute_2d_histogram_for_comparison, compute_3d_histogram_rgb, euclidean_distance, normalize_histogram
import os
import numpy as np

def test_euclidean_distance_1d_rgb():
    """
    Calculates the Euclidean distance between the 1D RGB histogram of a query image and other images in the test set, using normalized histograms.
    """
    test_dir = "tests/test_images"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No test images found!")
        return
    # Use the first image as query
    query_file = image_files[0]
    query_path = os.path.join(test_dir, query_file)
    query_img = load_image(query_path)
    query_hist = np.concatenate([
        compute_1d_histogram(query_img, 0),
        compute_1d_histogram(query_img, 1),
        compute_1d_histogram(query_img, 2)
    ])
    query_hist = normalize_histogram(query_hist)
    print(f"Query image: {query_file}\n")
    # Calculate distance to all other images
    results = []
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        img = load_image(image_path)
        hist = np.concatenate([
            compute_1d_histogram(img, 0),
            compute_1d_histogram(img, 1),
            compute_1d_histogram(img, 2)
        ])
        hist = normalize_histogram(hist)
        dist = euclidean_distance(query_hist, hist)
        results.append((image_file, dist))
    # Sort by distance
    results.sort(key=lambda x: x[1])
    print("Euclidean Distances 1D RGB (normalized):")
    for fname, dist in results:
        print(f"{fname:25s}  distance: {dist:.5f}")

def test_euclidean_distance_2d_rgb():
    """
    Calculates the Euclidean distance between the 2D RGB histograms of a query image and other images in the test set, using normalized histograms.
    """
    test_dir = "tests/test_images"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No test images found!")
        return
    query_file = image_files[0]
    query_path = os.path.join(test_dir, query_file)
    query_img = load_image(query_path)
    # Use all 2D RGB planes for comparison
    from image_processing import rgb_2d_all_planes_histogram_for_comparison
    query_hist = rgb_2d_all_planes_histogram_for_comparison(query_img, bins=32)
    query_hist = normalize_histogram(query_hist)
    results = []
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        img = load_image(image_path)
        hist = rgb_2d_all_planes_histogram_for_comparison(img, bins=32)
        hist = normalize_histogram(hist)
        dist = euclidean_distance(query_hist, hist)
        results.append((image_file, dist))
    results.sort(key=lambda x: x[1])
    print("\nEuclidean Distances 2D RGB (all planes, normalized):")
    for fname, dist in results:
        print(f"{fname:25s}  distance: {dist:.5f}")

def test_euclidean_distance_3d_rgb():
    """
    Calculates the Euclidean distance between the 3D RGB histograms of a query image and other images in the test set, using normalized histograms.
    """
    test_dir = "tests/test_images"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No test images found!")
        return
    query_file = image_files[0]
    query_path = os.path.join(test_dir, query_file)
    query_img = load_image(query_path)
    query_hist = compute_3d_histogram_rgb(query_img, bins=8)
    query_hist = normalize_histogram(query_hist)
    results = []
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        img = load_image(image_path)
        hist = compute_3d_histogram_rgb(img, bins=8)
        hist = normalize_histogram(hist)
        dist = euclidean_distance(query_hist, hist)
        results.append((image_file, dist))
    results.sort(key=lambda x: x[1])
    print("\nEuclidean Distances 3D RGB (normalized):")
    for fname, dist in results:
        print(f"{fname:25s}  distance: {dist:.5f}")

if __name__ == "__main__":
    test_euclidean_distance_1d_rgb()
    test_euclidean_distance_2d_rgb()
    test_euclidean_distance_3d_rgb() 