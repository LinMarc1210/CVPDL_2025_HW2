# CVPDL 2025 Homework 2: Long-tailed Object Detection

## Overview
This project implements object detection for drone images using a YOLO11s model train from scratch. It is part of the Computer Vision and Deep Learning course homework assignment. The goal is to detect and localize drone categories in provided image datasets.

## Features
- Custom dataset handling for drone images and annotations.
- Training pipeline with data augmentation and validation.
- Model fine-tuning using pre-trained YOLO on COCO dataset.
- Inference and visualization scripts for predictions.

## Installation

Set up the environment (please install `uv` first):
   - Using uv:
     ```bash
     uv sync
     ```
   - activate your environment
     ```bash
     .venv\Scripts\Activate.ps1 
     ```

   Required packages include:
   - ultralytics
   - PyTorch
   - Torchvision
   - Other dependencies as listed in `requirements.txt`

## Usage
**Note: please run the scripts in the `src` directory**
```bash
cd src
```

1. **Data Preparation**:
   - Run the data preparation script:
     ```bash
     python yolo_dataset.py
     ```
   - (Optional) Run the tuning script:
     ```bash
     python yolo_tune.py
     ```

2. **Training**:
   - Run the training script:
     ```bash
     python yolo_train.py
     ```
   - This will train the model, perform validation, and save checkpoints in `runs/detect/<model_name>` directory.
   - It will also save train and validation loss in `runs/detect/<model_name>/results.csv` directory.

3. **Inference**:
   - Use the inference script to run predictions on new images:
     ```bash
     python yolo_inference.py
     ```

## Contributing
This is a homework project for NTU_CVPDL_2025. For questions, refer to the course materials or contact the instructor.
