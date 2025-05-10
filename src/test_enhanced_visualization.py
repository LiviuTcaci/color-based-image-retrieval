# src/test_enhanced_visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing import load_image, convert_to_hsv

def display_enhanced_comparison(rgb_image, hsv_image, title="Comparație RGB vs HSV"):
    """
    Afișează o comparație detaliată între RGB și HSV, incluzând canalele individuale.
    """
    plt.figure(figsize=(15, 10))
    
    # RGB Image and channels
    plt.subplot(231)
    plt.imshow(rgb_image)
    plt.title('RGB Original')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(rgb_image[:,:,0], cmap='Reds')
    plt.title('Canal R')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(rgb_image[:,:,1], cmap='Greens')
    plt.title('Canal G')
    plt.axis('off')
    
    # HSV Image and channels
    plt.subplot(234)
    hsv_display = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    plt.imshow(hsv_display)
    plt.title('HSV')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(hsv_image[:,:,0], cmap='hsv')
    plt.title('Hue (H)')
    plt.axis('off')
    
    plt.subplot(236)
    plt.imshow(hsv_image[:,:,1], cmap='gray')
    plt.title('Saturation (S)')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    # Testează cu o imagine din setul de test
    image_path = "tests/test_images/astronaut.png"
    
    # Încarcă și convertește imaginea
    rgb_image = load_image(image_path)
    hsv_image = convert_to_hsv(rgb_image)
    
    # Afișează comparația îmbunătățită
    display_enhanced_comparison(rgb_image, hsv_image, 
                              f"Conversie RGB -> HSV: {image_path.split('/')[-1]}")

if __name__ == "__main__":
    main()