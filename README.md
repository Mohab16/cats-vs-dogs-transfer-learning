# Cats vs Dogs — Image Classification

A comprehensive binary image classification project that distinguishes cats from dogs using transfer learning with **MobileNetV2**. This repository provides dual implementations in **TensorFlow/Keras** and **PyTorch**, along with exploratory data analysis, training pipelines, evaluation tools, and a FastAPI inference server.

---

## Overview

This project demonstrates:
- **Transfer Learning**: Leverages pretrained MobileNetV2 for efficient feature extraction
- **Dual Framework Implementations**: Complete implementations in both TensorFlow/Keras and PyTorch for comparison
- **Modular Architecture**: Reusable components for dataset handling, model creation, training, evaluation, and inference
- **Production-Ready API**: FastAPI server for serving predictions on new images
- **Comprehensive Analysis**: EDA notebook and detailed training notebooks

---

## Quickstart

### 1. Setup Environment

```powershell
# Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your image dataset under `datasets/`:

```
datasets/
├── train/
│   ├── cats/
│   └── dogs/
└── validation/
    ├── cats/
    └── dogs/
```

### 3. Run Training & Evaluation

Open a notebook to train and evaluate:
- **TensorFlow**: [notebooks/Cats_vs_Dogs_Project_TensorFlow.ipynb](notebooks/Cats_vs_Dogs_Project_TensorFlow.ipynb)
- **PyTorch**: [notebooks/Cats_vs_Dogs_Project_PyTorch.ipynb](notebooks/Cats_vs_Dogs_Project_PyTorch.ipynb)

Or run the FastAPI inference server:

```powershell
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Project Structure

```
.
├── main.py                      # FastAPI inference server
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── notebooks/
│   ├── Cats_vs_Dogs_Project_TensorFlow.ipynb   # TensorFlow training & evaluation
│   ├── Cats_vs_Dogs_Project_PyTorch.ipynb      # PyTorch training & evaluation
│   └── EDA.ipynb                # Exploratory data analysis
│
├── datasets/
│   ├── vectorize.py            # Dataset preprocessing utilities
│   ├── train/
│   │   ├── cats/               # Training cat images
│   │   └── dogs/               # Training dog images
│   └── validation/
│       ├── cats/               # Validation cat images
│       └── dogs/               # Validation dog images
│
├── src/
│   ├── __init__.py
│   ├── TensorFlow_src/
│   │   ├── dataset.py          # Data loading, augmentation, preprocessing
│   │   ├── model.py            # MobileNetV2 transfer learning model
│   │   ├── train.py            # Training pipeline & fine-tuning
│   │   ├── evaluate.py         # Evaluation & visualization utilities
│   │   └── predictions.py      # Single-image inference helper
│   └── PyTorch_src/
│       ├── dataset.py          # PyTorch dataset & dataloader utilities
│       ├── model.py            # MobileNetV2 PyTorch implementation
│       ├── train.py            # Training & optimization loop
│       ├── evaluate.py         # Metrics & confusion matrix
│       └── predictions.py      # Inference utilities
│
├── models/
│   ├── head_training.keras     # TensorFlow: classification head model
│   ├── head_training.pth       # PyTorch: classification head checkpoint
│   ├── final_training.keras    # TensorFlow: fine-tuned full model
│   └── fine_tuning.pth         # PyTorch: fine-tuned full model
│
└── testing_images/             # Sample images for testing predictions
```

---

## Framework Implementations

### TensorFlow/Keras Implementation

**File**: `src/TensorFlow_src/`

- **Model Architecture**: MobileNetV2 base + GlobalAveragePooling2D + Dropout(0.2) + Dense(1, sigmoid)
- **Training Approach**: Two-stage training (head training → full model fine-tuning)
- **Data Augmentation**: Built-in preprocessing with TensorFlow data pipelines

**Usage**:

```python
from src.TensorFlow_src.model import create_model
from src.TensorFlow_src.train import train_head_model, fine_tune_model
from src.TensorFlow_src.predictions import predict_image

# Create model
model = create_model()

# Train classification head
model, history = train_head_model()

# Fine-tune full model
final_model, final_history = fine_tune_model(initial_epoch=history.epoch[-1])

# Predict on new image
label, confidence = predict_image('models/final_training.keras', 'path/to/image.jpg')
```

### PyTorch Implementation

**File**: `src/PyTorch_src/`

- **Model Architecture**: MobileNetV2 features + AdaptiveAvgPool2d + Dropout(0.2) + Linear(1280, 1)
- **Training Approach**: Modular training loop with configurable epochs and learning rates
- **Device Support**: Automatic GPU/CPU detection with CUDA support

**Usage**:

```python
from src.PyTorch_src.model import CatsDogsModel, load_trained_model
from src.PyTorch_src.train import train_model
from src.PyTorch_src.predictions import predict_image

# Create model
model = CatsDogsModel()

# Train model
train_model(model, train_loader, num_epochs=10)

# Load trained model
model = load_trained_model('models/fine_tuning.pth')

# Predict on new image
label, confidence = predict_image(model, 'path/to/image.jpg')
```

---

## API Server (FastAPI)

The `main.py` file provides a REST API for inference.

### Endpoints

**GET** `/`
- Returns: `{"message": "Cats vs Dogs API is running!"}`

**POST** `/predict`
- **Input**: Image file (JPEG/PNG) via multipart/form-data
- **Output**: JSON with label and confidence
- **Example**:

```json
{
  "label": "Cat",
  "confidence": 0.92,
  "probability": 0.08
}
```

### Running the Server

```powershell
# Install FastAPI and Uvicorn
pip install fastapi uvicorn

# Start server
uvicorn main:app --reload

# Interactive API docs: http://localhost:8000/docs
# ReDoc documentation: http://localhost:8000/redoc
```

---

## Training & Evaluation

### Training Workflow

Both frameworks follow a similar two-stage training approach:

1. **Stage 1: Classification Head Training**
   - Freeze MobileNetV2 base layers
   - Train only the classification head
   - Typical duration: 5-10 epochs

2. **Stage 2: Fine-tuning**
   - Unfreeze some base layers (optional)
   - Train with lower learning rate
   - Typical duration: 10-20 epochs

### Evaluation Metrics

- Accuracy
- Precision & Recall
- Confusion Matrix
- Loss curves

### Reproducing Results

```python
# TensorFlow
from src.TensorFlow_src.evaluate import evaluate_model, plot_confusion_matrix
loss, acc = evaluate_model('models/final_training.keras', validation_dataset)
plot_confusion_matrix('models/final_training.keras', validation_dataset)

# PyTorch
from src.PyTorch_src.evaluate import evaluate_model
loss, acc = evaluate_model(model, validation_loader, device)
```

---

## Requirements

- **Python**: 3.8+
- **Core Dependencies**:
  - TensorFlow >= 2.12.0
  - PyTorch (for PyTorch implementation)
  - NumPy >= 1.23
  - Pillow >= 9.0
  - Scikit-learn >= 1.0
  - Matplotlib >= 3.5
  - JupyterLab >= 3.0
  - FastAPI (for API server)
  - Uvicorn (for running FastAPI)

Install all dependencies:

```powershell
pip install -r requirements.txt
```

---

## Model Artifacts

### TensorFlow Models
- `models/head_training.keras` — Classification head model
- `models/final_training.keras` — Fine-tuned complete model (recommended for inference)

### PyTorch Models
- `models/head_training.pth` — Classification head checkpoint
- `models/fine_tuning.pth` — Fine-tuned complete model (recommended for inference)

---

## Exploratory Data Analysis

Open [notebooks/EDA.ipynb](notebooks/EDA.ipynb) to explore:
- Dataset distribution (cats vs dogs)
- Image statistics and preprocessing effects
- Sample visualizations

---

## Next Steps & Ideas

- 📈 **Scale Dataset**: Increase dataset size and improve class balance
- 🔄 **Ensemble Methods**: Combine TensorFlow and PyTorch models
- 🚀 **Model Export**: Convert models to ONNX or TensorFlow Lite for mobile/edge deployment
- 📱 **Mobile App**: Build iOS/Android app using exported models
- ☁️ **Cloud Deployment**: Deploy FastAPI server on AWS/GCP/Azure
- 🔍 **Advanced Architectures**: Try EfficientNet or Vision Transformer (ViT) backbones
- 📊 **Monitoring**: Add model performance tracking and inference logging

---

## Contributing

Feel free to fork, modify, and improve this project!

---

## License

This project is open source and available under the MIT License.


