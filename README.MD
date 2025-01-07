# Waste Classifier

This project consists of a system that can classify waste in two categories: recyclable and organic. We train and test multiple models and evaluate them, pick the best one for our purpose and integrate it in a system that can classify on images, videos and live-video using device camera.

## Table of Contents

- [Waste Classifier](#waste-classifier)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Features

- train models for waste classification
- evaluate models
- check results of each models
- use models

## Installation

1. Clone the repository:

   ```bash

   git clone https://github.com/MarinelaAdam/wasteclassification.git
2. Create the virtual environment:

   ```python
    python -m venv venv
3. Activate it:

    ```bash
    .\\venv\\Scripts\\activate
4. Install dependencies:

   ```bash
   pip install -r requirements.txt

## Directory Structure

A detailed breakdown of the project's structure:

```plaintext
Project_CV_Waste/
├── Datasets/             # Contains datasets used for model training and testing
├── Live_samples/         # Example images/videos for testing the model
├── Models/               # Saved model checkpoints and training data (loss & accuracy)
├── Results/              # Evaluation metrics of the models
├── trackings/            # Processed videos/images saved here
├── arhitecture.py        # Custom model architecture definition
├── custom_eval.py        # Evaluation for larger custom models
├── custom_models.py      # Training for larger custom models
├── CustomSmallModel/     # Training and evaluation for smaller models MobileNets (not v4)
├── mobilenetv4.py        # Training and evaluation for MobileNetV4
├── requirements.txt      # List of project dependencies
├── trainvsval_custom.py  # Plots accuracy and loss during training (for custom models)
├── trainvsval.py         # Plots accuracy and loss during training (for own architecture)
├── useModel_complex.py   # The most accurate usage of model (single object)
├── useModel_simple.py    # Smaller version for single-object detection (lower inference)
├── useModel.py           # Recommended script for multi-object detection (best inference)
├── WasteClasifier_eval/  # Evaluation for the own model
├── WasteClasifier/       # Training for the own model
├── README.md             # Project documentation
└── yolo11x.pt            # YOLOv11X model checkpoint
```

## Usage

1. Prepare your environment by following the Installation steps.
2. Choose which version to run: useModel.py/useModel_simple.py/useModel_complex.py
3. Make the changes before main for (single image, provided video, live video)
4. For useModel you can check results also in trackings.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
