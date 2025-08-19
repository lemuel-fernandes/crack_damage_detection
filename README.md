# Building Crack Damage Detection

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent computer vision system for automatic detection and classification of structural damage in building images, including cracks, potholes, and surface damage.

![Analysis Demo](docs/analysis_example.png)

## Features

- **Automated Damage Detection**: Identifies various types of structural damage using advanced computer vision algorithms
- **Multi-type Classification**: Distinguishes between:
  - Long Cracks
  - Short Cracks  
  - Potholes
  - Surface Damage
- **Severity Assessment**: Categorizes damage into Low, Medium, and High severity levels
- **Comprehensive Analysis**: Provides detailed metrics including area, perimeter, aspect ratio, and circularity
- **Visual Output**: Generates annotated images and step-by-step processing visualizations
- **Detailed Reports**: Exports analysis results in multiple formats

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

```bash
pip install opencv-python numpy matplotlib
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/lemuel-fernandes/crack_damage_detection.git
cd crack_damage_detection
```

2. Install dependencies:
```bash
pip install opencv-python numpy matplotlib
```

3. Place your building image as `building.jpg` in the project directory

4. Run the analysis:
```bash
python crack_detection.py
```

## Usage

### Basic Usage

The system is designed to work with a file named `building.jpg` by default:

```python
python crack_detection.py
```

### Programmatic Usage

```python
from crack_detection import CrackDetectionSystem

# Initialize detector
detector = CrackDetectionSystem()

# Analyze image
result = detector.process_image("path/to/your/image.jpg")

# Access results
print(f"Found {result['total_damages']} damage areas")
for damage in result['damages']:
    print(f"{damage['type']}: {damage['severity']} severity")
```

### Custom Configuration

```python
config = {
    'gaussian_kernel_size': (5, 5),
    'canny_low_threshold': 50,
    'canny_high_threshold': 150,
    'min_contour_area': 100,
    'crack_length_threshold': 100,
    # ... other parameters
}

detector = CrackDetectionSystem(config=config)
```

## Output

The system generates several outputs in the `building_analysis_results/` directory:

- **Annotated Image**: Original image with damage areas highlighted and labeled
- **Analysis Visualization**: 6-panel figure showing each processing step
- **Text Report**: Detailed findings with recommendations
- **Console Output**: Real-time analysis summary

### Sample Output

```
BUILDING DAMAGE ANALYSIS SUMMARY
============================================
Image: building.jpg
Dimensions: 1920 x 1080 pixels
Total Damage Areas Found: 5

Detailed Findings:
----------------------------------------
 1. Long Crack - High Severity
    Location: (450, 300)
    Size: 150 x 8 pixels
    Area: 1200 pixels

 2. Short Crack - Medium Severity
    Location: (720, 180)
    Size: 80 x 5 pixels
    Area: 400 pixels

Summary by Type:
  Long Crack: 2
  Short Crack: 3

Summary by Severity:
  High: 1
  Medium: 3
  Low: 1

Recommendations:
⚠️  URGENT: 1 high-severity damage(s) need immediate attention!
📋 Schedule repair for 3 medium-severity damage(s)
📝 Monitor 1 low-severity damage(s) for changes
```

## Algorithm Overview

The detection system uses a multi-stage computer vision pipeline:

1. **Preprocessing**
   - Gaussian blur for noise reduction
   - Bilateral filtering for edge preservation
   - CLAHE for contrast enhancement

2. **Edge Detection**
   - Canny edge detection
   - Sobel gradient detection
   - Edge map combination

3. **Morphological Processing**
   - Closing operations to connect nearby edges
   - Opening operations to remove noise

4. **Feature Extraction**
   - Contour detection and filtering
   - Geometric property analysis

5. **Classification & Severity Assessment**
   - Rule-based damage type classification
   - Multi-factor severity evaluation

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gaussian_kernel_size` | (5, 5) | Gaussian blur kernel size |
| `canny_low_threshold` | 50 | Canny edge detection low threshold |
| `canny_high_threshold` | 150 | Canny edge detection high threshold |
| `min_contour_area` | 100 | Minimum contour area to consider |
| `max_contour_area` | 50000 | Maximum contour area to consider |
| `crack_length_threshold` | 100 | Minimum length for long crack classification |
| `min_aspect_ratio` | 2.0 | Minimum aspect ratio for crack detection |

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## Requirements

Create a `requirements.txt` file with:

```
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Machine learning-based classification
- [ ] Support for batch processing multiple images
- [ ] Web interface for easy usage
- [ ] Mobile app integration
- [ ] Real-time video analysis
- [ ] 3D damage mapping

## Troubleshooting

### Common Issues

**Image not found error**:
- Ensure `building.jpg` is in the same directory as the script
- Check file permissions

**Import errors**:
- Verify all dependencies are installed: `pip list`
- Update pip: `pip install --upgrade pip`

**Poor detection results**:
- Try adjusting threshold parameters in the configuration
- Ensure good image quality and lighting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- NumPy and Matplotlib for numerical computing and visualization
- Contributors and testers who helped improve the system

## Contact

For questions, suggestions, or collaboration opportunities:

- GitHub: [@lemuel-fernandes](https://github.com/lemuel-fernandes)
- Repository: [crack_damage_detection](https://github.com/lemuel-fernandes/crack_damage_detection)

---

**⭐ Star this repository if you find it helpful!**
