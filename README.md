# Color-based Image Retrieval

Un sistem de căutare și recuperare a imaginilor bazat pe culori, implementat în Python.

## Descriere
Acest proiect implementează un sistem de căutare a imaginilor bazat pe caracteristicile culorilor, utilizând:
- Histograme 1D, 2D și 3D în spațiile de culoare RGB și HSV
- Reprezentare grafică a histogramelor
- Compararea vectorilor de caracteristici folosind distanța euclidiană

## Instalare
1. Clonează repository-ul:
```bash
git clone https://github.com/LiviuTcaci/color-based-image-retrieval.git
cd color-based-image-retrieval
```

2. Creează și activează un mediu virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # Pentru Unix/macOS
```

3. Instalează dependențele:
```bash
pip install -r requirements.txt
```

## Utilizare
```bash
python src/main.py
```

## Testare
```bash
python src/test_performance.py
```

## Tehnologii Utilizate
- Python 3.12
- OpenCV pentru procesarea imaginilor
- NumPy pentru calcule numerice
- Matplotlib pentru vizualizare
- Pillow pentru manipularea imaginilor
- scikit-image pentru funcții avansate de procesare a imaginilor