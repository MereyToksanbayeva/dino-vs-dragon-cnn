Dino vs Dragon – CNN Classifier

A professional deep-learning project built using TensorFlow/Keras.
This model classifies images into two categories:

Dino

Dragon

It demonstrates a complete ML pipeline: dataset preparation, CNN creation, training, evaluation, visualization, and model saving.

📂 Project Structure
dino-vs-dragon-cnn/
│
├── dino_vs_dragon_cnn.py          # Training script
├── dino_vs_dragon_model.h5        # Saved CNN model
├── dino_dragon_dataset/           # Dataset folder
│     ├── train/
│     │     ├── dino/
│     │     └── dragon/
│     └── val/
│           ├── dino/
│           └── dragon/
└── README.md

🚀 Overview

This project uses a Convolutional Neural Network (CNN) to distinguish between two image categories: dinosaurs and dragons.

It demonstrates:

Binary image classification

Data preprocessing

Building a CNN architecture

Training & validation

Accuracy visualization

Saving a TensorFlow model

Perfect for ML portfolio, internships, and Erasmus+ applications.

🛠 Technologies Used

Python 3.10

TensorFlow / Keras

Matplotlib

Pillow

ImageDataGenerator

Convolutional Neural Networks (CNN)

📦 Installation

Install necessary libraries:

pip install tensorflow keras pillow matplotlib


Run the training script:

python dino_vs_dragon_cnn.py

🧠 Model Architecture

The model consists of:

3× Convolutional layers (32, 64, 128 filters)

3× MaxPooling layers

Flatten layer

Dense(128) layer

Dense(1) with sigmoid activation

This is a classic architecture for binary classification.

📊 Training Results

During training, the script displays:

Training Accuracy

Validation Accuracy

Example graph:

Accuracy
↑
│       ┌──────── val_acc
│   ┌───┘
│ ┌─┘
│┌┘
└──────────────→ Epochs


(High accuracy is expected with small sample dataset.)

📁 Dataset

The dataset is structured as:

train/dino
train/dragon
val/dino
val/dragon


You can replace the images with real data for more realistic results.

🚀 Future Improvements

Possible enhancements:

Use real Dino vs Dragon dataset

Add data augmentation

Add dropout layers

Use transfer learning (MobileNetV2, VGG16)

Deploy with Flask REST API

Convert to TensorFlow Lite for Android

📝 Author

Merey Toksanbayeva
AI / ML Student & Developer
GitHub: https://github.com/MereyToksanbayeva

⭐ Erasmus+ Ready

This project meets European ML portfolio standards and is suitable for:

Erasmus+ Traineeship

ML/AI internship applications

University research labs

Computer vision portfolios
