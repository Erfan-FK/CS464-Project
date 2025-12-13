# Project Scripts & Usage Guide

## Scripts Overview

*   **`data/data_prep.py`**: Handles data splitting (train/val/test) and computes class weights to handle class imbalance. It generates indices in `data/splits/`.
*   **`data/extract_features.py`**: Runs a pre-trained ResNet18 (without the final classification layer) on the dataset to extract 512-dimensional feature vectors for every image. Saves these features to `.npz` files in `features/`.
*   **`CNN/train.py`**: A sample script that trains a simple Convolutional Neural Network (CNN) from scratch using the raw image data and the generated splits.
*   **`Random_Forest/train.py`**: Trains a scikit-learn RandomForest on the precomputed feature `.npz` files and saves metrics/model artifacts.

## Environment Setup

1.  Ensure you have Conda installed.
2.  Create the environment using the provided `environment.yml`:
    ```bash
    conda env create -f environment.yml
    ```
3.  Activate the environment:
    ```bash
    conda activate plant_env
    ```

### Alternative Setup (pip / Google Colab)

If you do not have Conda (e.g., on Google Colab), you can install the required packages using pip:

```bash
pip install torch torchvision numpy scikit-learn pandas matplotlib pillow tqdm
```

## Data Preparation

### 1. Extract Raw Data
If the data is not yet extracted, use the following command to extract the tarball:
```bash
tar -xvf data/house_plant_species_raw/house_plant_species.tar -C data/
```

### 2. Splits & Feature Extraction
*Note: These steps are likely already completed. Check `data/splits` and `features/`.*

If you need to regenerate splits or features:
```bash
# Generate splits
python data/data_prep.py --data_root data/house_plant_species --splits_dir data/splits

# Extract features (requires splits to exist)
python data/extract_features.py --data_root data/house_plant_species --splits_dir data/splits --out_dir features/

# Train Random Forest on features
python Random_Forest/train.py --features_dir features --out_dir Random_Forest/output
```

Since feature extraction is already complete, you do **not** need to load raw images or perform computationally expensive preprocessing. You can load the pre-computed feature vectors directly.

### Step-by-Step Guide

1.  **Create a new directory** for your model (e.g., `SVM/`, `RandomForest/`).
2.  **Create a training script** (e.g., `SVM/train_svm.py`).
3.  **Load the data**: Use `numpy.load` to read the `.npz` files from the `features/` directory.
4.  **Train & Evaluate**: Use Scikit-Learn or other libraries to train on the loaded arrays.

### Example Code Snippet
Here is a template for a new training script (e.g., `SVM/train.py`):

```python
import sys
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "features"

def load_features(split_name):
    """Load X and y from .npz file."""
    file_path = FEATURES_DIR / f"{split_name}_features.npz"
    data = np.load(file_path)
    return data['X'], data['y']

def main():
    print("Loading features...")
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")
    X_test, y_test = load_features("test")

    print(f"Train shape: {X_train.shape}")
    
    # 2. Initialize Model
    # Example: Support Vector Machine
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    
    # 3. Train
    print("Training SVM...")
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("Evaluating...")
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
```
