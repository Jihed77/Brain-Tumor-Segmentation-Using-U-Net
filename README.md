# 🧠 Brain Tumor Segmentation Using U‑Net

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)  
*(Replace with your actual Streamlit Cloud URL after deployment)*

## 📌 Overview
This project implements a **U‑Net** convolutional neural network to automatically segment low‑grade gliomas (brain tumors) from MRI scans.  
The model is trained on the public [LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) and achieves **85.8% Dice coefficient** on unseen test images.

## 🚀 Features
- **Interactive web app** built with [Streamlit](https://streamlit.io/) – upload an MRI, get the predicted tumor mask instantly.
- **Data augmentation** (rotation, shift, zoom, flip) to improve generalization.
- **Monitor metrics** – Intersection over Union (IoU) and Dice coefficient (F1‑score).
- **Pre‑trained model** – ready to use for inference.

## 📊 Dataset
- **Source:** [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Kaggle)
- **Content:** 3,929 brain MRI images + corresponding ground‑truth masks (tumor vs background)
- **Split:**  
  - Training: 3,545 images  
  - Validation: 187 images  
  - Test: 197 images

## 🏗️ Model Architecture
The U‑Net follows the classic encoder‑decoder design with skip connections:

| Encoder (down) | Decoder (up) |
|----------------|---------------|
| 64 → 128 → 256 → 512 → 1024 | 512 → 256 → 128 → 64 |
| 2x Conv2D each level | 2x Conv2D each level |
| MaxPooling (2×2) | Conv2DTranspose (2×2) |
| - | Concatenation with corresponding encoder feature maps |

Final layer: 1×1 convolution + sigmoid activation.

## 📈 Results

| Metric       | Validation | Test  |
|--------------|------------|-------|
| **Dice**     | 0.8575     | 0.8575 |
| **IoU**      | 0.7515     | 0.7515 |
| **Loss** (binary cross‑entropy) | 0.0037     | 0.0037 |

## 🛠️ Local Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jihed77/Brain-Tumor-Segmentation-Using-U-Net.git
   cd Brain-Tumor-Segmentation-Using-U-Net
