# Project: Tomato Leaf Disease Detection using YOLO26n

Fine-tuning a custom YOLO26 nano model for agricultural research.

## Overview

This repository contains an end-to-end Jupyter Notebook pipeline for fine-tuning a YOLO26 nano model (`yolo26n.pt`) to detect and classify tomato leaf diseases. The implementation is tailored for research environments, using PyTorch and Ultralytics with hardware acceleration.

## Final Evaluation Results

Evaluated on the unbiased test set:
* **Test mAP50-95:** 0.9222
* **Test mAP50:** 0.9700
* **Test Precision:** 0.9607
* **Test Recall:** 0.9320

## Requirements & Environment

* **Python:** 3.13.9
* **PyTorch:** `2.13.0+cu126`
* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA enabled)[cite: 1]
* **Framework:** Ultralytics YOLO[cite: 1]

## Project Structure

```text
├── Tomato Leaf Disease.v63i.yolo26/    # Dataset directory (train, valid, test splits & data.yaml)
├── runs/                             # Training logs, weights, and evaluation outputs
└── Tomato-leaf-disease-detection.ipynb # Main research pipeline notebook
