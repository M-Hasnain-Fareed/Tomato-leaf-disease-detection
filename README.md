# Project: Tomato Leaf Disease Detection using YOLO26n



Fine-tuning a custom YOLO26 nano model for agricultural research.

## Overview

This repository contains an end-to-end Jupyter Notebook pipeline for fine-tuning a YOLO26 nano model (`yolo26n.pt`) to detect and classify tomato leaf diseases. The implementation is tailored for research environments, using PyTorch and Ultralytics with hardware acceleration.

## Requirements & Environment

* **Python:** 3.13.9


* **PyTorch:** `2.13.0+cu126`

* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA enabled)


* **Framework:** Ultralytics YOLO



## Project Structure

```text
├── Tomato Leaf Disease.v63i.yolo26/    # Dataset directory (train, valid, test splits & data.yaml)
├── runs/                             # Training logs, weights, and evaluation outputs
└── Tomato-leaf-disease-detection.ipynb # Main research pipeline notebook

```

## Pipeline Workflow

1. **Environment Verification:** Checks PyTorch version, CUDA availability, and GPU specifications.


2. **Dataset Configuration:** Locates and validates the `data.yaml` configuration file structure.


3. **Model Initialization:** Loads the pretrained `yolo26n.pt` baseline architecture for transfer learning.


4. **Fine-Tuning:** Executes training using research-grade hyperparameters (e.g., 50 epochs, batch size of 4, image size 640).


5. **Validation & Metrics:** Evaluates the best-performing weights (`best.pt`) on the validation split to compute quantitative metrics (`mAP50-95`, `mAP50`, Precision, Recall).


6. **Inference & Visualization:** Runs prediction on test samples and displays performance curves (training progression, confusion matrix, Precision-Recall curve).



## Usage

1. Clone the repository and ensure your dataset is placed in the root directory following the expected Ultralytics structure.
2. Open the notebook in VS Code or Jupyter Lab:
```bash
jupyter notebook Tomato-leaf-disease-detection.ipynb

```


3. Run the cells sequentially to reproduce the training and evaluation workflow.
