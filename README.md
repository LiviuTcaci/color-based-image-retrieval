# Color-based Image Retrieval

Un sistem de căutare și recuperare a imaginilor bazat pe culori, implementat în Python.

## Descriere
Acest proiect implementează un sistem de căutare a imaginilor bazat pe caracteristicile culorilor, utilizând:
- Histograme 2D și 3D în spațiile de culoare RGB și HSV
- Reprezentare grafică a histogramelor
- Compararea vectorilor de caracteristici folosind distanța euclidiană

## Structura Proiectului
```
color-based-image-retrieval/
├── src/
│   ├── image_processing/
│   │   ├── histogram.py      # Implementarea histogramelor 2D și 3D
│   │   ├── color_spaces.py   # Conversii între spațiile de culoare RGB și HSV
│   │   └── comparison.py     # Metode de comparare a vectorilor de caracteristici
│   ├── visualization/
│   │   └── plots.py         # Funcții pentru vizualizarea histogramelor
│   └── main.py             # Punctul de intrare al aplicației
├── tests/
│   └── test_images/        # Imagini pentru testare
├── docs/
│   ├── presentation.pptx   # Prezentarea proiectului
│   └── documentation.pdf   # Documentația detaliată
├── requirements.txt        # Dependențele proiectului
└── README.md              # Acest fișier
```

## Instalare
1. Clonează repository-ul:
```bash
git clone https://github.com/YourUsername/color-based-image-retrieval.git
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

## Tehnologii Utilizate
- Python 3.12
- OpenCV pentru procesarea imaginilor
- NumPy pentru calcule numerice
- Matplotlib pentru vizualizare
- scikit-image pentru funcții avansate de procesare a imaginilor