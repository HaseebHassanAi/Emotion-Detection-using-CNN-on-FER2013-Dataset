# Emotion Detection using CNN on FER2013 Dataset

This project implements a **Convolutional Neural Network (CNN)** with **L1/L2 regularization, Batch Normalization, Dropout, and Data Augmentation** to classify human emotions from facial images using the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

---

## 🚀 Features
- Custom **CNN architecture** with:
  - Convolutional + MaxPooling layers
  - **Batch Normalization**
  - **Dropout** for regularization
  - **L2 regularization** to reduce overfitting
- **Data augmentation** for robust training
- **EarlyStopping** and **ReduceLROnPlateau** callbacks
- Trained on **7 emotion classes**:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Supports prediction on **new images**

---

## 📂 Dataset
We use the **FER2013** dataset, available on Kaggle:
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

It consists of **48x48 grayscale facial images** labeled into 7 emotion categories.

---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/emotion-detection-fer2013.git
cd emotion-detection-fer2013
pip install -r requirements.txt
