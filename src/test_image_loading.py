from image_processing import load_image, convert_to_hsv, get_image_info
import os

def test_image_loading():
    # Test images directory
    test_dir = "tests/test_images"
    
    # Test each image in the directory
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nTesting image: {image_file}")
            
            try:
                # Load the image
                rgb_image = load_image(image_path)
                print("Image successfully loaded in RGB")
                
                # Convert to HSV
                hsv_image = convert_to_hsv(rgb_image)
                print("HSV conversion successful")
                
                # Display image information
                rgb_info = get_image_info(rgb_image)
                hsv_info = get_image_info(hsv_image)
                
                print("\nRGB Information:")
                for key, value in rgb_info.items():
                    print(f"  {key}: {value}")
                
                print("\nHSV Information:")
                for key, value in hsv_info.items():
                    print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    test_image_loading()