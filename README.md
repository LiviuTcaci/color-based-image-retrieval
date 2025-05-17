# Color-based Image Retrieval

A color-based image search and retrieval system implemented in Python.

## Description
This project implements an image search system based on color features, using:
- 1D, 2D, and 3D histograms in RGB and HSV color spaces
- Graphical representation of histograms
- Feature vector comparison using Euclidean distance
- Graphical interface for visualization and search

## Features
- **Histogram Visualization**:
  - 1D RGB and HSV histograms
  - 2D RGB and HSV histograms (all planes)
  - 3D RGB and HSV histograms (scatter plot visualization)
- **Similar Image Search**:
  - Search based on 1D RGB/HSV histograms
  - Search based on 2D RGB/HSV histograms
  - Search based on 3D RGB/HSV histograms
  - Combined search (all histogram types)
- **Performance Testing**:
  - Measurement of computation time for each histogram type
  - Performance comparison between different methods
  - Testing with multiple image sets

## Installation
1. Clone the repository:
```bash
git clone https://github.com/LiviuTcaci/color-based-image-retrieval.git
cd color-based-image-retrieval
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # For Unix/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the application:
```bash
python src/main.py
```

2. In the graphical interface:
   - Select an image using the "Select Image" button
   - View histograms using buttons in the "RGB Visualization" and "HSV Visualization" sections
   - Search for similar images by selecting the histogram type and clicking "Search Similar Images"

## Testing
1. Test performance:
```bash
python src/test_performance.py
```

2. Run individual tests:
```bash
python src/test_histogram_normalization.py
python src/test_histogram_precision.py
python src/test_euclidean_distance.py
python src/test_search.py
```

## Project Structure
- `src/main.py` - Main application with graphical interface
- `src/image_processing.py` - Image processing and histogram computation functions
- `src/test_*.py` - Test files for different components
- `tests/test_images/` - Directory containing test images

## Technologies Used
- Python 3.12
- OpenCV for image processing
- NumPy for numerical computations
- Matplotlib for visualization
- Pillow for image manipulation
- scikit-image for advanced image processing functions
- Tkinter for graphical interface