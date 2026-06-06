# 🚀 Industrial CNN Classifier

**Machine vision applications are very common in the automotive industry**. They are commonly solved by implementing vision systems and developing rule-based applications (e.g., thresholding, filtering, and contour detection). But what happens when an application has a lot of variations in light or position? These classic tools are not enough; this is why **AI plays such a big role in machine vision.**

---

## 🧠 Convolutional Neural Networks
CNNs are widely used in computer vision because they automatically extract features and have proven a better performance compared to ML models.
I wanted to **simplify** and **accelerate** the **process of training a CNN** by developing a **modular architecture** with custom Python modules to train different CNN architectures leveraging **transfer learning**.
- In order to deploy a CNN model, a machine vision engineer would only need to capture some samples in order to start training a CNN, divide the samples by classes and execute a simple jupyter notebook.
- In many real-world industrial applications, this pipeline has demonstrated robustness and positive results against industrial environment variations and complex applications.

---

## 🐍 Python Modules
The project contains 3 main custom libraries to simplify the training process:
1. `dataset.py`: A custom module made to create and split datasets given an input folder containing images of each class.
2. `models.py`: This module provides different pre-trained CNN architectures to leverage the model as a **feature extractor**.
3. `trainer.py`: As part of the modular architecture, this library is in charge of the entire training process, controlling the full workflow and hyperparameter definition (epochs, optimizer, criterion, etc).
- 💡 `Note:` **Please refer to the Jupyter notebook (.ipynb file) for a detailed overview of the full workflow.**

---

## 📁 Project Structure
```text
├── datasets/              # This folder contains the sample images (train/test) in order to train. Each class must be separated by folders. 
│   └── my_application/
│       ├── test_images
│       └── train_images
├── outputs/
│   ├── models
│   └── plots
├── src/                   # Source code
│   ├── __init__.py
│   ├── augmentation.py    # Module to perform image augmentation, given an input images folder.
│   ├── dataset.py         # Creates and splits datasets into train and validation.
│   ├── models.py          # Contains different pre-trained CNN architectures (ResNet18).
│   └── trainer.py         # Performs the full training workflow.
└── housing-classification.ipynb
└── o-ring-inspection.ipynb
└── welding-check.ipynb
└── README.md
```

---

## ➡ Next Steps
- Development of a user-friendly GUI to control the full workflow.
- Integrate more CNN architectures: ConvNeXt, EfficientNet, DenseNet,etc.
