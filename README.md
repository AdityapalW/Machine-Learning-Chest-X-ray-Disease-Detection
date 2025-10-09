# Machine-Learning-Chest-X-ray-Disease-Detection
A deep learning project for automated chest X-ray analysis using CNNs. It predicts the likelihood of thoracic diseases like pneumonia, cardiomegaly, and emphysema from NIH Chest X-ray images. Built with PyTorch/TensorFlow for preprocessing, training, and evaluation to support AI-based medical screening.

# Directory Structure
```text
project-root/
│
│   README.md              # Project overview and instructions
│   requirements.txt       # Python dependencies
│   .gitignore             # Files ignored by Git
│
├── data/
│   ├── raw/               # Unmodified source data
│   ├── processed/         # Preprocessed/cleaned data
│   └── external/          # Third-party or external sources
│
│
├── src/
│   ├── preprocessing/     # Scripts for cleaning, augmenting, and prepping data
│   ├── models/            # Model architectures, CNN here
│   ├── training/          # Training and validation routines
│   ├── evaluation/        # Metrics and analysis
│   └── utils.py           # Helper functions
│
├── outputs/
│   ├── models/            # Saved trained weights
│   ├── results/           # Figures, logs, or predictions
│   └── logs/              # Logging files
│
└── tests/                 # Unit tests (optional)