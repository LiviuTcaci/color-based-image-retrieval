from image_processing import load_image, convert_to_hsv, convert_to_rgb, display_rgb_hsv_comparison
import os
import numpy as np

def test_color_conversion():
    test_dir = "tests/test_images"
    
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nTesting conversion for: {image_file}")
            
            try:
                rgb_image = load_image(image_path)
                hsv_image = convert_to_hsv(rgb_image)
                rgb_converted = convert_to_rgb(hsv_image)
                
                mse = np.mean((rgb_image - rgb_converted) ** 2)
                print(f"Mean Squared Error: {mse:.2f}")
                
                display_rgb_hsv_comparison(rgb_image, hsv_image, 
                                         f"RGB <-> HSV Conversion: {image_file}")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    test_color_conversion() 