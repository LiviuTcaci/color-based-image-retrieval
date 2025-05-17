from image_processing import analyze_rgb_distribution, display_2d_hsv_histograms
import os

def test_2d_histograms():
    test_dir = "tests/test_images"
    
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nAnalyzing 2D histogram for: {image_file}")
            
            try:
                analyze_rgb_distribution(image_path)
                rgb_image = analyze_rgb_distribution.__globals__['load_image'](image_path)
                display_2d_hsv_histograms(rgb_image, f"2D HSV Histograms: {image_file}")
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_2d_histograms() 