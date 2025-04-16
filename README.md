# Text-to-Satellite Image Segmentation

This project implements a UNet-based model for satellite image segmentation using natural language descriptions.

### Prerequisites
- Python 3.12.x
- PyTorch and related packages:
  ```bash
  pip3 install torch torchvision torchaudio
  pip3 install pillow numpy matplotlib albumentations
  ```

### Terrain Classes
| Color | Class | Label |
|-------|-------|-------|
| Cyan | Urban | 0 |
| Yellow | Agriculture | 1 |
| Magenta | Rangeland | 2 |
| Green | Forest | 3 |
| Blue | Water | 4 |
| White | Barren | 5 |
| Black | Unknown | 6 |

## 1. Image Classifier (UNet)

### Features
- UNet architecture for image segmentation
- 7-class terrain classification 
- RGB satellite image input (256x256)
- PyTorch implementation

### Model Architecture
The UNet model is configured with:
- Input: RGB images (3 channels)
- Output: 7 terrain classes
- Feature maps: [64, 128, 256, 512]

## 2. NLP Text Processing

### Features
- TF-IDF based text processing
- Interactive and batch processing modes
- Query history tracking
- Confidence scoring for matches

### Usage
1. Install required packages:
   ```bash
   pip install scikit-learn pandas numpy joblib
   ```

2. Interactive Terminal Mode:
   ```bash
   python process_queries.py
   ```
   - Enter natural language queries
   - Get immediate results with confidence scores
   - History saved to `data-nlp/inputHistory.txt` and `data-nlp/outputHistory.txt`

3. Batch Processing Mode:
   ```bash
   python batch_process.py input.in output.out
   ```
   - Process multiple queries from input file
   - Results saved to output file
   - Simple JSON format for integration

### Input Examples
```
"show the forest areas"
"highlight urban regions"
"display water bodies"
"mark agricultural zones"
```

### Output Format
```json
{
    "label": "forest",
    "color": "green",
    "confidence": 0.85
}
```

## 3. GitHub Guide

### Initial Setup
1. Install Git from [git-scm.com](https://git-scm.com/)
2. Clone the repository:
   ```bash
   git clone https://github.com/BigBrothersen/Text-to-Satellite_Image_Segmentation
   cd Text-to-Satellite_Image_Segmentation
   ```

### Development Workflow
1. Sync with main branch:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Create and switch to feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make changes and commit:
   ```bash
   git add .
   git commit -m "descriptive commit message"
   git push origin feature/your-feature-name
   ```

## Contributors
- Kenneth Saputra Limanto (122040029)
- Kevin William (122040033)
- William Hansen Loe (122040046)
- Nemtsov Vladimir (124100014)