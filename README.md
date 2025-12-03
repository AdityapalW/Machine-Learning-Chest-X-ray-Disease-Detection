# Machine-Learning-Chest-X-ray-Disease-Detection
A deep learning project for automated chest X-ray analysis using CNNs. It predicts the likelihood of thoracic diseases like pneumonia, cardiomegaly, and emphysema from NIH Chest X-ray images. Built with PyTorch/TensorFlow for preprocessing, training, and evaluation to support AI-based medical screening.


## Directory Structure
```text
project-root/
│
│   README.md                      # Project overview and instructions
│   requirements.txt               # Python dependencies
│   .gitignore                     # Files ignored by Git
│
├── data/
│   └── raw/                       # Unmodified source data
│        ├── images/               # chest X-ray images
│        └── sample_labels.scv     # Labels CSV
│
└── src/
    ├── preprocessing/             # Scripts for cleaning, augmenting, and prepping data
    ├── models/                    # Model architectures, CNN here
    ├── training/                  # Training and validation routines
    ├── evaluation/                # Metrics and analysis
    └── utils.py                   # Helper functions
```


## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Usage](#usage)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)


## Installation

You can install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```
The main dependencies are:
- torch, torchvision
- numpy
- pandas
- scikit-learn
- Pillow
- tqdm
- streamlit (for the web app)

## Dataset

This project is designed around the NIH Chest X-ray dataset. Due to storage limitation we used [Random Sample of NIH Chest X-ray Dataset from Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/sample?resource=download)

## Model Architecture
The project uses a custom convolutional neural network (CNN) defined in [`src/models/custom_cnn.py`](src/models/custom_cnn.py) as [`src.models.custom_cnn.CustomCNN`](src/models/custom_cnn.py).

- **Input**: 224×224 RGB chest X‑ray images
- **Backbone**: 3 convolutional blocks  
  - `Conv2d → BatchNorm2d → ReLU → MaxPool2d` with 32, 64 and 128 filters
- **Classifier**:  
  - Flatten  
  - `Linear(128×28×28 → 256) → ReLU → Dropout(0.5)`  
  - `Linear(256 → 14)` (one logit per disease)
- **Task**: multi‑label classification over 14 thoracic disease classes using `BCEWithLogitsLoss` (see [`src/training/loss.get_loss_function`](src/training/loss.py)).


## Usage
#### Running the Model

1. Clone the repository:

   ```bash
   git clone https://github.com/AdityapalW/Machine-Learning-Chest-X-ray-Disease-Detection.git
   ```

2. Download the Kaggle NIH Chest X-ray Data and Extract it in the `data/raw` directory:

   ```bash
   https://www.kaggle.com/datasets/nih-chest-xrays/sample?resource=download
   ```

3. Navigate into the directory:

   ```bash
   cd Machine-Learning-Chest-X-ray-Disease-Detection
   ```

4. Start the training:

   ```bash
   python main.py
   ```

5. Evaluate the model:

   ```bash
   python visualize.py
   ```

## Training and Evaluation

Training is done from [`main.py`](main.py):

1. **Data split & preprocessing**
   - The label CSV is split into train/validation/test sets using [`src.datasets.utils.split_dataset`](src/datasets/utils.py).
   - Channel‑wise mean and std are computed on the training split with [`src.datasets.utils.compute_mean_std`](src/datasets/utils.py).
   - Image transforms and augmentations are created via [`src.preprocessing.preprocess.get_transforms`](src/preprocessing/preprocess.py):
     - Train: resize to 224×224, random flips/rotations/affine, color jitter, normalization.
     - Val/Test: resize to 224×224 + normalization only.

2. **Model & loss**
   - The model is [`src.models.custom_cnn.CustomCNN`](src/models/custom_cnn.py) with 14 outputs (one per disease).
   - Loss: binary cross‑entropy with logits via [`src.training.loss.get_loss_function`](src/training/loss.py) (`BCEWithLogitsLoss`).

3. **Training loop**
   - Implemented in [`src.training.train.train_model`](src/training/train.py).
   - For each epoch:
     - Train on the training set (optionally with mixed precision).
     - Track training loss and accuracy.
     - Evaluate on the validation set and save a checkpoint each epoch.

4. **Final evaluation**
   - After training, the best model is evaluated on the test split using [`src.evaluation.evaluate.evaluate_model`](src/evaluation/evaluate.py), which:
     - Computes average loss.
     - Applies `sigmoid` + 0.5 threshold to get multi‑label predictions.
     - Reports overall accuracy across all labels.


## Results
After training, the model achieved the following performance metrics:

- **Accuracy**: 
- **Precision**: 
- **Recall**: 
- **AUC**: 
