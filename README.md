
# 📸 Yelp Photos Image Classification

This project aims to classify images from the **Yelp Photos dataset** into one of five categories: `food`, `inside`, `outside`, `drink`, and `menu`, using a Convolutional Neural Network (CNN) built with **Keras/TensorFlow**.

---

## Problem Statement

The goal is to develop a robust image classification model that accurately predicts the **label of a given Yelp photo** using deep learning.

---

## 📂 Dataset

- **Source**: [Yelp Open Dataset](https://www.yelp.com/dataset)
- **Images**: > 200,000 labeled photos
- **Labels**:
  - `food`
  - `inside`
  - `outside`
  - `drink`
  - `menu`

---

## Data Preprocessing

Performed on initial raw dataset to make it model-ready:

1. **Sampling**: 10,000 samples per class (or all if <10k)
2. **Resizing**: All images resized to **128x128 pixels**
3. **Augmentation (Train Set)**:
   - Random crop
   - Horizontal flip
   - Color jitter (brightness, contrast, saturation)
4. **Advanced Preprocessing** :
   - Gaussian blur
   - Histogram equalization
   - Intensity thresholding
5. **Train/Val/Test Split**: 80/10/10 split, **without leakage**
6. **Structured Directory**:



data_sorted/
├── train/
├── val/
└── test/





## Model Architecture

### CNN Model (Final tuned version)

python
Input: (128, 128, 3)

Conv2D(32, 3x3, ReLU) + MaxPool
Conv2D(64, 3x3, ReLU) + MaxPool
Conv2D(128, 3x3, ReLU) + MaxPool

Flatten
Dense(128, ReLU) + L2 Regularization + Dropout(0.3)
Dense(5, Softmax)


Loss: Categorical Crossentropy

Optimizer: RMSprop (lr=0.0005)

Regularization: L2 + Dropout

Callbacks:

EarlyStopping

ReduceLROnPlateau



📊 Evaluation Metrics

Accuracy: Train, Validation, and Test

Precision, Recall, F1-Score

Confusion Matrix

AUC Score: Macro & per class

ROC Curves

Latest Model Results (Test Set)
Metric	Score
Accuracy	78.20%
Macro AUC	~0.88
Best Class	food
Weakest Class	drink (Recall was low)


Key Improvements Over Time

Attempt	Change Made	Effect
v1	Basic CNN	Overfitting, poor generalization
v2	L2 Regularization, Dropout	Reduced overfitting
v3	RMSprop, ReduceLROnPlateau	Better validation performance
v4	Class Weights	Boosted underrepresented classes
v5	Data Augmentation (Flow)	Boosted robustness, reduced overfitting

Future Work
Implement transfer learning.

Try auto-tuning hyperparameters (Optuna, Keras Tuner)

Deploy as an API or web interface using Flask or FastAPI

Run inference on real-time Yelp images

📁 Repository Structure

├── data/
├── metadata/
├── models/
├── notebooks/
├── utils/
├── evaluate.py
├── train.py
└── README.md


Requirements

Python 3.10+

TensorFlow / Keras

Pandas, NumPy, Matplotlib, Scikit-learn

GPU (recommended for training)

Credits
Yelp Open Dataset

TensorFlow & Keras

Google Cloud Platform (GCP)