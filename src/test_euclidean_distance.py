from image_processing import load_image, compute_1d_histogram, compute_2d_histogram_for_comparison, compute_3d_histogram_rgb, euclidean_distance, normalize_histogram
import os
import numpy as np

def test_euclidean_distance_1d_rgb():
    """
    Calculează distanța Euclidiană între histograma 1D RGB a unei imagini query și celelalte imagini din setul de test, folosind histograme normalizate.
    """
    test_dir = "tests/test_images"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Nu există imagini de test!")
        return
    # Folosește prima imagine ca query
    query_file = image_files[0]
    query_path = os.path.join(test_dir, query_file)
    query_img = load_image(query_path)
    query_hist = np.concatenate([
        compute_1d_histogram(query_img, 0),
        compute_1d_histogram(query_img, 1),
        compute_1d_histogram(query_img, 2)
    ])
    query_hist = normalize_histogram(query_hist)
    print(f"Imagine query: {query_file}\n")
    # Calculează distanța față de toate celelalte imagini
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
    # Sortează după distanță
    results.sort(key=lambda x: x[1])
    print("Distanțe Euclidiene 1D RGB (normalizate):")
    for fname, dist in results:
        print(f"{fname:25s}  distanță: {dist:.5f}")

def test_euclidean_distance_2d_rgb():
    test_dir = "tests/test_images"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Nu există imagini de test!")
        return
    query_file = image_files[0]
    query_path = os.path.join(test_dir, query_file)
    query_img = load_image(query_path)
    # Folosește toate planurile 2D RGB pentru comparație
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
    print("\nDistanțe Euclidiene 2D RGB (toate planurile, normalizate):")
    for fname, dist in results:
        print(f"{fname:25s}  distanță: {dist:.5f}")

def test_euclidean_distance_3d_rgb():
    test_dir = "tests/test_images"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Nu există imagini de test!")
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
    print("\nDistanțe Euclidiene 3D RGB (normalizate):")
    for fname, dist in results:
        print(f"{fname:25s}  distanță: {dist:.5f}")

if __name__ == "__main__":
    test_euclidean_distance_1d_rgb()
    test_euclidean_distance_2d_rgb()
    test_euclidean_distance_3d_rgb() 