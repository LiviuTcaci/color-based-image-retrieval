import os
import time
import numpy as np
from image_processing import load_image, convert_to_hsv, compute_1d_histogram, compute_2d_histogram_for_comparison, compute_3d_histogram_rgb, compute_3d_histogram_hsv, euclidean_distance, normalize_histogram, compute_2d_histogram_hsv

def compute_1d_histogram_hsv(hsv_image, channel, bins=256):
    return np.histogram(hsv_image[:, :, channel].flatten(), bins=bins, range=(0, 256))[0]

def performance_test():
    test_dir = "tests/test_images"
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Nu există imagini de test!")
        return
    print(f"Număr imagini de test: {len(image_files)}\n")
    # 1D RGB
    t0 = time.time()
    hists_1d_rgb = []
    for img_path in image_files:
        img = load_image(img_path)
        h = np.concatenate([
            compute_1d_histogram(img, 0),
            compute_1d_histogram(img, 1),
            compute_1d_histogram(img, 2)
        ])
        hists_1d_rgb.append(h)
    t1 = time.time()
    print(f"Timp calcul histograme 1D RGB: {t1-t0:.4f} secunde")
    # 1D HSV
    t0 = time.time()
    hists_1d_hsv = []
    for img_path in image_files:
        img = load_image(img_path)
        hsv = convert_to_hsv(img)
        h = np.concatenate([
            compute_1d_histogram_hsv(hsv, 0, bins=180),
            compute_1d_histogram_hsv(hsv, 1, bins=256),
            compute_1d_histogram_hsv(hsv, 2, bins=256)
        ])
        hists_1d_hsv.append(h)
    t1 = time.time()
    print(f"Timp calcul histograme 1D HSV: {t1-t0:.4f} secunde")
    # 2D RGB (toate planurile)
    t0 = time.time()
    hists_2d_rgb = []
    for img_path in image_files:
        img = load_image(img_path)
        h = np.concatenate([
            compute_2d_histogram_for_comparison(img, 0, 1, bins=32).flatten(),
            compute_2d_histogram_for_comparison(img, 0, 2, bins=32).flatten(),
            compute_2d_histogram_for_comparison(img, 1, 2, bins=32).flatten()
        ])
        hists_2d_rgb.append(h)
    t1 = time.time()
    print(f"Timp calcul histograme 2D RGB (toate planurile): {t1-t0:.4f} secunde")
    # 2D HSV (toate planurile)
    t0 = time.time()
    hists_2d_hsv = []
    for img_path in image_files:
        img = load_image(img_path)
        hsv = convert_to_hsv(img)
        h = np.concatenate([
            compute_2d_histogram_hsv(hsv, 0, 1, 180, 256).flatten(),
            compute_2d_histogram_hsv(hsv, 0, 2, 180, 256).flatten(),
            compute_2d_histogram_hsv(hsv, 1, 2, 256, 256).flatten()
        ])
        hists_2d_hsv.append(h)
    t1 = time.time()
    print(f"Timp calcul histograme 2D HSV (toate planurile): {t1-t0:.4f} secunde")
    # 3D RGB
    t0 = time.time()
    hists_3d_rgb = []
    for img_path in image_files:
        img = load_image(img_path)
        h = compute_3d_histogram_rgb(img, bins=8).flatten()
        hists_3d_rgb.append(h)
    t1 = time.time()
    print(f"Timp calcul histograme 3D RGB: {t1-t0:.4f} secunde")
    # 3D HSV
    t0 = time.time()
    hists_3d_hsv = []
    for img_path in image_files:
        img = load_image(img_path)
        h = compute_3d_histogram_hsv(img, bins=(18, 16, 16)).flatten()
        hists_3d_hsv.append(h)
    t1 = time.time()
    print(f"Timp calcul histograme 3D HSV: {t1-t0:.4f} secunde")
    # Toate RGB (1D+2D+3D)
    t0 = time.time()
    hists_all_rgb = [normalize_histogram(np.concatenate([h1, h2, h3])) for h1, h2, h3 in zip(hists_1d_rgb, hists_2d_rgb, hists_3d_rgb)]
    for i in range(len(hists_all_rgb)):
        for j in range(i+1, len(hists_all_rgb)):
            _ = euclidean_distance(hists_all_rgb[i], hists_all_rgb[j])
    t1 = time.time()
    print(f"Timp comparație distanțe Euclidiene Toate RGB (toate perechile): {t1-t0:.4f} secunde")
    # Toate HSV (1D+2D+3D)
    t0 = time.time()
    hists_all_hsv = [normalize_histogram(np.concatenate([h1, h2, h3])) for h1, h2, h3 in zip(hists_1d_hsv, hists_2d_hsv, hists_3d_hsv)]
    for i in range(len(hists_all_hsv)):
        for j in range(i+1, len(hists_all_hsv)):
            _ = euclidean_distance(hists_all_hsv[i], hists_all_hsv[j])
    t1 = time.time()
    print(f"Timp comparație distanțe Euclidiene Toate HSV (toate perechile): {t1-t0:.4f} secunde")
    # Toate combinate (1D+2D+3D RGB+HSV)
    t0 = time.time()
    hists_all_combined = [normalize_histogram(np.concatenate([h1, h2])) for h1, h2 in zip(hists_all_rgb, hists_all_hsv)]
    for i in range(len(hists_all_combined)):
        for j in range(i+1, len(hists_all_combined)):
            _ = euclidean_distance(hists_all_combined[i], hists_all_combined[j])
    t1 = time.time()
    print(f"Timp comparație distanțe Euclidiene Toate combinate (toate perechile): {t1-t0:.4f} secunde")

if __name__ == "__main__":
    performance_test() 