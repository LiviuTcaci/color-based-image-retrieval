import os
import cv2
import numpy as np
from skimage import data
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

def create_test_images():
    test_dir = Path("tests/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    sk_images = {
        'astronaut': data.astronaut(),
        'camera': data.camera(),
        'coffee': data.coffee(),
        'hubble_deep_field': data.hubble_deep_field(),
        'moon': data.moon(),
        'chelsea': data.chelsea(),
        'rocket': data.rocket()
    }
    
    for name, img in sk_images.items():
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(str(test_dir / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved {name}.png")

    extra_images = {
        'immunohistochemistry': data.immunohistochemistry(),
        'coins': data.coins(),
        'text': data.text(),
        'clock': data.clock(),
        'grass': data.grass()
    }
    for name, img in extra_images.items():
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(str(test_dir / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved {name}.png")

    astro = data.astronaut()
    crop = astro[30:180, 60:210]
    cv2.imwrite(str(test_dir / "astronaut_crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print("Saved astronaut_crop.png")

    url_images = {
        'cat': 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg',
        'dog': 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg',
        'paris': 'https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg',
        'mountain': 'https://upload.wikimedia.org/wikipedia/commons/6/6f/Alpspix_Mountain_View.jpg',
        'flower': 'https://upload.wikimedia.org/wikipedia/commons/4/40/Flower_pink_rose.jpg',
        'car': 'https://upload.wikimedia.org/wikipedia/commons/7/7f/Classic-car-01.jpg',
        'fruit': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg',
        'city': 'https://upload.wikimedia.org/wikipedia/commons/5/5c/NYC_Midtown_Skyline_at_night_-_Jan_2006_edit1.jpg'
    }
    for name, url in url_images.items():
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.save(str(test_dir / f"{name}.jpg"))
            print(f"Downloaded and saved {name}.jpg")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

if __name__ == "__main__":
    create_test_images()