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
  pip install tqdm pandas numpy joblib scikit-learn
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
- BERT-based text processing with semantic similarity matching
- Support for compound queries (e.g., "show forests and urban areas")
- Confidence-based thresholding (0.90 - 0.70)
- Dynamic color assignment based on natural language input
- Interactive and batch processing modes
- Query history tracking and logging
- Model retraining capability

### Usage

1. Interactive Terminal Mode:
   ```bash
   cd NLP
   python process_queries.py
   ```
   Features:
   - Real-time query processing
   - Confidence score display
   - Multiple label support
   - History logging

2. Batch Processing Mode:
   ```bash
   cd NLP
   python batch_process.py input.in output.out
   ```
   Features:
   - Process multiple queries
   - JSON output format
   - Automatic threshold adjustment
   - Compound query support

3. Color Mapping Generation:
   ```bash
   python colorMap.py
   ```
   Features:
   - Interactive query input
   - Dynamic color reassignment
   - Maintains label indices
   - Outputs formatted color mapping

### Input/Output Examples

1. Input Query:
   ```
   show forests green and urban areas yellow
   ```

2. NLP Output (`1.out`):
   ```json
   {"label": "forest", "color": "green"}
   {"label": "urban", "color": "yellow"}
   ```

3. Color Mapping (`result.out`):
   ```
   (255, 255, 0): 0    # Urban
   (0, 0, 0): 1    # Agriculture
   (0, 0, 0): 2    # Rangeland
   (0, 255, 0): 3    # Forest
   (0, 0, 255): 4    # Water
   (0, 0, 0): 5    # Barren
   (0, 0, 0): 6    # Unknown
   ...
   ```

### File Structure
```
NLP/
├── models/              # BERT model files
├── data-nlp/
│   ├── csv/            # Training data
│   └── logs/           # Query history
├── NLP.py              # Core NLP processor
├── batch_process.py    # Batch processing
├── process_queries.py  # Interactive mode
└── create.py          # Training data generator
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