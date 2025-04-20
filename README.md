# Text-to-Satellite Image Segmentation

This project implements a UNet-based model for satellite image segmentation using natural language descriptions.

### Prerequisites
- Python 3.12.x
- PyTorch and related packages:
  ```bash
  pip install torch torchvision torchaudio
  pip install pillow numpy matplotlib albumentations
  ```
- NLP related packages:
  ```bash
  pip install transformers torch sentence-transformers
  pip install tqdm pandas numpy joblib
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
- BERT-based text processing for improved semantic understanding
- Support for compound queries (e.g., "show forests and urban areas")
- Interactive and batch processing modes
- Query history tracking
- Confidence scoring for matches
- Model retraining capability

### Usage
1. Interactive Terminal Mode:
   ```bash
   python process_queries.py
   ```
   - Enter natural language queries
   - Get immediate results with confidence scores
   - Supports multiple labels in single query
   - History saved to `data-nlp/logs/inputHistory.txt` and `data-nlp/logs/outputHistory.txt`

2. Batch Processing Mode:
   ```bash
   python batch_process.py input.in output.out
   ```
   - Process multiple queries from input file
   - Results saved to output file
   - Supports compound queries
   - Simple JSON format for integration

### Input Examples
```
"show forests and urban areas"
"highlight water bodies and agriculture"
"display desert regions"
"mark agricultural zones and forests"
```

### Output Format
```json
[
    {
        "label": "forest",
        "color": "green"
    },
    {
        "label": "urban",
        "color": "cyan"
    }
]
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