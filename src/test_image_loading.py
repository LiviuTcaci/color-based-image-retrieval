from image_processing import load_image, convert_to_hsv, get_image_info
import os

def test_image_loading():
    # Directorul cu imagini de test
    test_dir = "tests/test_images"
    
    # Testează fiecare imagine din director
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nTestare imagine: {image_file}")
            
            try:
                # Încarcă imaginea
                rgb_image = load_image(image_path)
                print("✓ Imagine încărcată cu succes în RGB")
                
                # Convertește în HSV
                hsv_image = convert_to_hsv(rgb_image)
                print("✓ Conversie în HSV reușită")
                
                # Afișează informații despre imagine
                rgb_info = get_image_info(rgb_image)
                hsv_info = get_image_info(hsv_image)
                
                print("\nInformații RGB:")
                for key, value in rgb_info.items():
                    print(f"  {key}: {value}")
                
                print("\nInformații HSV:")
                for key, value in hsv_info.items():
                    print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"✗ Eroare: {str(e)}")

if __name__ == "__main__":
    test_image_loading()