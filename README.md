# Cats vs Dogs — Image Classification (MobileNetV2 transfer learning)

This repository implements a binary image classifier that distinguishes cats from dogs using transfer learning with MobileNetV2 (TensorFlow / Keras). The codebase is modular and reproducible via the notebook and `src/` utilities.

## Quickstart

1. Create and activate a Python virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare the dataset under `datasets/` as:

```
datasets/train/cats/
datasets/train/dogs/
datasets/validation/cats/
datasets/validation/dogs/
```

3. Reproduce the analysis and training by opening the notebook: [notebooks/Cats_vs_Dogs_Project.ipynb](notebooks/Cats_vs_Dogs_Project.ipynb)

## Project Structure

- `notebooks/` — walkthrough and reproducible notebook
- `datasets/` — training and validation image folders
- `src/` — reusable modules:
  - `dataset.py` — data loading, splitting, augmentation
  - `model.py` — model creation and preprocessing
  - `train.py` — training and fine-tuning helpers
  - `evaluate.py` — evaluation and plotting utilities
  - `predictions.py` — single-image prediction helper
- `models/` — saved model artifacts (`head_training.keras`, `final_training.keras`)
- `requirements.txt` — pinned Python packages

## Reproduce: training & evaluation

- Train the classification head:

```python
from src.train import train_head_model
model, history = train_head_model()
```

- Fine-tune the combined model:

```python
from src.train import fine_tune_model
final_model, final_history = fine_tune_model(initial_epoch=history.epoch[-1])
```

- Evaluate a saved model:

```python
from src.evaluate import evaluate_model, plot_confusion_matrix
loss, acc = evaluate_model('../models/final_training.keras', validation_dataset)
plot_confusion_matrix('../models/final_training.keras', validation_dataset)
```

## Inference

Use the helper in `src/predictions.py` for single-image predictions. Example in the notebook demonstrates calling `predict_image(model_path, img_path)`.

## Requirements

- Python 3.8+ recommended
- See `requirements.txt` for exact versions. Install with `pip install -r requirements.txt`.

## Artifacts

- Trained models: `models/head_training.keras`, `models/final_training.keras`
- Notebook: `notebooks/Cats_vs_Dogs_Project.ipynb`

## Next steps & ideas

- Increase dataset size and class balance
- Try stronger backbones or ensemble models
- Export to TensorFlow Lite for mobile/edge
- Add a simple Flask/FastAPI inference endpoint for serving predictions


