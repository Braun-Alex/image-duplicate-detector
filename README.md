# 📸 Image Duplicate Detector 📸

## 🕵️ About the Project 🕵️

This project dives into the use of **Convolutional Neural Networks (CNNs)** to discover similarities between images. A **Siamese Neural Network** has been implemented using Python and TensorFlow, powered by twin **ConvNeXt networks**. These networks work in tandem, comparing feature vectors through the lens of **cosine similarity**. The training process is enhanced by the **contrastive loss** function, ensuring our model learns to distinguish between duplicates with finesse.

### 🌟 Highlights 🌟:

- Utilization of **Siamese Neural Network** architecture.
- Comparison of images using **cosine similarity**.
- Training with **contrastive loss** for optimal learning.

## 🛠 Installation 🛠

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/Braun-Alex/image-duplicate-detector.git
cd image-duplicate-detector
pip3 install -r requirements.txt
```

## 🚀 How to Run the Program 🚀

The program operates in an interactive mode, offering you the flexibility to train, test, and utilize machine learning models seamlessly.

To start the program, run:

```bash
cd image-duplicate-detector
python3 main.py
```
