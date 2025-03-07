# Optical Sorting Code

This repository provides Python scripts for **16-bit image capturing**, **real-time preview** (black-and-white and color modes), and a basic framework for **optical sorting** using Thorlabs cameras and OpenCV. It also includes an end-to-end workflow that guides you through capturing a background image, tuning parameters, building a training set, training a model, and running classification.

---

## Contents

- **BW_PREVIEW.py**  
  Demonstrates grayscale (black-and-white) preview and associated capture functions.

- **colourful_PREVIEW.py**  
  Demonstrates color preview and capture functions.

- **Helper Functions** (within scripts)  
  - `analyze_frame(frame)`: Prints basic info about the frame (shape, dtype, min/max, mean).  
  - `show_16bit_with_matplotlib(frame_16u)`: Displays a 16-bit image (grayscale) using matplotlib.  
  - `capture_single_image(...)`: Captures a single 16-bit image, saves as `.npy` (raw data) and `.png`.  
  - `capture_continuously(...)`: Continuously captures images at a specified frame rate, optionally applying real-time processing or classification.  
  - `detect_rice_by_subtraction(...)`: Demonstrates a pixel-difference method for detecting foreground objects (like rice), then draws contours.  
  - `label_contours(...)`: Labels each detected contour with a unique index for manual inspection.  
  - `get_training_set(...)`: Interactively records user feedback (normal/abnormal) for each detected contour, building a labeled dataset.  
  - `train_model(...)`: Uses the dataset to train a RandomForest classifier and saves the model.  
  - `classification(...)`: Applies the trained model to each contour in the captured frame, drawing bounding boxes in green/red.

---

## Requirements

1. **Python 3.x** (tested with 3.7+).
2. **pylablib** (for Thorlabs camera control).
3. **OpenCV** (for image processing).
4. **matplotlib** (for visualization; optional but recommended).
5. **numpy** (for numerical operations).
6. **scikit-learn** (e.g., `RandomForestClassifier`, if you use the classification logic).
7. **joblib** (for saving/loading the trained model).

