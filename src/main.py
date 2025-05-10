from image_processing import load_image, convert_to_hsv
from histograms import compute_2d_histogram, compute_3d_histogram
from visualization import plot_2d_histogram, plot_3d_histogram

def main():
    # Exemplu de utilizare
    image_path = "tests/test_images/astronaut.png"
    
    # Încărcare imagine
    rgb_image = load_image(image_path)
    hsv_image = convert_to_hsv(rgb_image)
    
    # Calculare histograme
    rgb_hist_2d = compute_2d_histogram(rgb_image)
    hsv_hist_2d = compute_2d_histogram(hsv_image)
    
    rgb_hist_3d = compute_3d_histogram(rgb_image)
    hsv_hist_3d = compute_3d_histogram(hsv_image)
    
    # Vizualizare
    plot_2d_histogram(rgb_hist_2d, "Histograma 2D RGB")
    plot_2d_histogram(hsv_hist_2d, "Histograma 2D HSV")
    
    plot_3d_histogram(rgb_hist_3d, "Histograma 3D RGB")
    plot_3d_histogram(hsv_hist_3d, "Histograma 3D HSV")

if __name__ == "__main__":
    main() 