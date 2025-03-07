"""
Created on 2025/2/19  14:11

@author: YANG FENG RUI
"""
# -*- coding: utf-8 -*-
"""
camera_capture.py

Function Description:
1. capture_single_image() function: Open the camera, capture a single image, and save in 16-bit format.
2. capture_continuously() function: Continuously capture at the specified frame rate, saving each as 16-bit .npy.
3. Includes several helper functions (analyze_frame, show_16bit_with_matplotlib) which can be used optionally.

Usage:
- Open this script in PyCharm and run it directly. Then in main() you can call the functions you need.
- Or in another script, do: from camera_capture import capture_single_image / capture_continuously to use them.
"""

import os
import time
import numpy as np
import pylablib as pll
from pylablib.devices import Thorlabs
import matplotlib.pyplot as plt
import cv2

# Initialize (can check available devices, etc.)
pll.list_backend_resources("serial")


def show_16bit_with_matplotlib(frame_16u):
    """
    Use matplotlib to display a 16-bit image.
    frame_16u: dtype=uint16, shape=(height, width) or (height, width, channels).
    """
    plt.figure()
    plt.imshow(frame_16u, cmap='gray', vmin=0, vmax=65535)
    plt.colorbar(label='Pixel value')
    plt.title("16-bit image (Matplotlib)")
    plt.show()


def analyze_frame(frame):
    """
    Print some basic statistical information of the frame.
    """
    print("Shape:", frame.shape)
    print("Dtype:", frame.dtype)
    print("Min:", frame.min())
    print("Max:", frame.max())
    print("Mean:", frame.mean())


def capture_single_image(name='background', exposure_time=0.04):
    """
    Function: Open the camera, set exposure, capture a single image, save it as a 16-bit .npy and PNG, then close the camera.

    Usage:
      from camera_capture import capture_single_image
      capture_single_image()
    """
    # 1. Open the camera
    cam = Thorlabs.ThorlabsTLCamera()

    # 2. (Optional) Set ROI
    # width, height = 1440, 1080
    # cam.set_roi(0, width, 0, height, hbin=1, vbin=1)

    # 3. Set exposure time (unit: seconds), for example 10 ms
    cam.set_exposure(exposure_time)

    # 4. Start acquisition (nframes=1 means only capture one frame)
    cam.setup_acquisition(nframes=1)
    cam.start_acquisition()

    try:
        # 5. Wait and read the image
        cam.wait_for_frame(timeout=10.0)
        frame = cam.read_newest_image()
        analyze_frame(frame)

        # 6. Visualization (can be commented out)
        # show_16bit_with_matplotlib(frame)

        # 7. Save as .npy (16-bit raw data)
        np.save(name + '.npy', frame)
        print(f"[INFO] 16-bit image saved as: {name}.npy")

        # 8. Also save as a 16-bit PNG (OpenCV supports 16-bit depth)
        cv2.imwrite('single_frame_16bit.png', frame)
        print("[INFO] 16-bit image saved as: single_frame_16bit.png")

    finally:
        # 9. Close the camera
        cam.close()
        print("[INFO] Camera closed.")


def capture_continuously(func, model, bg_gray=[0],
                         min_area=2500,
                         morph_kernel_size=3,
                         threshold_val=4,
                         exposure_time=0.05, frame_num=10):
    """
    Continuously capture images, call 'func' to process them (e.g., classification),
    then display in real time the result_img (8-bit image with bounding boxes/prediction) returned by func.
    """

    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(exposure_time)

    cam.setup_acquisition(nframes=frame_num)
    cam.start_acquisition()

    frame_count = 0
    last_time = time.time()
    last_frame_time = last_time
    file_index = 0

    print(f"[INFO] Start continuous capture, {frame_num} frames per second.")

    try:
        while True:
            now = time.time()

            # If enough time has passed (>=1/frame_num), capture a frame
            if (now - last_frame_time) >= 1.0 / frame_num:
                frame = cam.read_newest_image()
                if frame is None:
                    continue

                # ---- Call your processing function (e.g., classification) ----
                flag, result_img = func(
                    str(file_index),
                    model,
                    bg_gray,
                    frame,  # Original 16-bit grayscale
                    min_area,
                    morph_kernel_size,
                    threshold_val
                )

                # If flag==0 is returned, stop capturing
                if flag == 0:
                    print("[INFO] func returned 0, stopping loop.")
                    break

                # ---- Real-time display: now result_img is an 8-bit BGR image with bounding boxes ----
                cv2.imshow("Live Preview", result_img)

                # Handle key to exit (ESC = 27)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] ESC pressed, exiting loop.")
                    break

                # Record count, update time
                file_index += 1
                frame_count += 1
                last_frame_time = now

            # Every 1 second, calculate the actual FPS
            if (now - last_time) >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"[INFO] Actual FPS: {fps:.2f}")
                frame_count = 0
                last_time = now

    except KeyboardInterrupt:
        print("[INFO] Caught Ctrl+C, preparing to close camera.")
    finally:
        cam.close()
        cv2.destroyAllWindows()  # Close the window
        print("[INFO] Camera closed, script ended.")


def i(base_name, bg_gray, sample, min_area, morph_kernel_size, threshold_val):
    answer_bg = input("Do you need to capture a background image? (y/n): ").strip().lower()
    print(answer_bg)
    return 0


import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_rice_by_subtraction(
        bg_gray,
        sample_gray,
        min_area=1000,
        morph_kernel_size=2,
        threshold_val=4
):
    """
    Use pixel-difference method to detect rice.
    Returns:
      final_image: the result image (if you need color display, drawn in red) based on sample_img
      fg_mask: binary mask
      valid_contours: list of valid contours
    """
    # 1. Read the background grayscale
    # bg_gray = np.load(bg_gray_path)

    # 2. Read the current frame (already grayscale)
    # sample_gray = np.load(sample_path, allow_pickle=True)

    # 3. Perform difference
    diff = cv2.absdiff(bg_gray, sample_gray)

    # 4. Binarize
    _, fg_mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

    # 5. Morphological denoising and hole-filling
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. Find contours
    fg_mask = fg_mask.astype(np.uint8)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Filter small contours by area and draw them on the image
    #    If you need to draw “red” on a grayscale image, you must convert to BGR first.
    final_image = cv2.cvtColor(sample_gray, cv2.COLOR_GRAY2BGR)
    tmp = final_image.astype(np.float32)

    # Normalize to [0,1]
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())

    # Multiply to [0,255]
    tmp = tmp * 255

    # Finally convert to uint8
    final_image = tmp.astype(np.uint8)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        valid_contours.append(cnt)
        # Draw in red (0,0,255) in BGR
        cv2.drawContours(final_image, [cnt], -1, (0, 0, 255), 2)

    return final_image, fg_mask, valid_contours

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # Set Chinese font to "SimHei" or others
matplotlib.rcParams['axes.unicode_minus'] = False     # Properly display minus sign

def label_contours(image, contours):
    """
    Label each contour in the given single-channel grayscale image with a unique index.
    """
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Only need a single gray value (e.g., 255 means white)
        cv2.putText(
            image, str(i), (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, 255, 2, cv2.LINE_AA
        )
    return image


def get_training_set(base_name, nothing, bg_gray, sample, min_area, morph_kernel_size, threshold_val):
    """
    1. Perform difference to get contours (grayscale)
    2. Label on the image
    3. Interactively let the user decide whether it’s “normal rice” or “incorrect contour” for removal
    4. Save the results
    """

    # 1. Detect rice contours (grayscale difference)
    #    Make sure detect_rice_by_subtraction is also grayscale-based
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction(
        bg_gray=bg_gray,
        sample_gray=sample,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )

    # 2. Label indices (for user reference)
    labeled_image = label_contours(final_image, valid_contours)

    # 3. Display the image (grayscale) using Matplotlib
    # Display window
    cv2.namedWindow('Labeling Window', cv2.WINDOW_NORMAL)
    # Set window size, for example 800x600
    cv2.resizeWindow('Labeling Window', 800, 600)
    cv2.imshow('Labeling Window', labeled_image)

    normal_flags = []
    filtered_contours = []

    for i, cnt in enumerate(valid_contours):
        # Ensure window events are handled, otherwise it may freeze
        cv2.waitKey(1)

        user_input = input(f"Is contour {i} normal rice? (y/n/e): ").lower().strip()

        if user_input in ["y", "yes"]:
            normal_flags.append(True)
            filtered_contours.append(cnt)
        elif user_input in ["n", "no"]:
            normal_flags.append(False)
            filtered_contours.append(cnt)
        elif user_input in ["e", "error"]:
            print(f"Contour {i} has been removed.")
        else:
            print(f"Invalid input {user_input}, removing contour.")

    cv2.destroyAllWindows()

    # 6. Store contours and labels
    #    Only keep the filtered_contours
    conts_as_arrays = [cnt.reshape(-1, 2) for cnt in filtered_contours]
    data_dict = {
        "is_normal": normal_flags,
        "contours": conts_as_arrays,
        "data": sample  # Optionally keep the original image
    }

    # Assume sample_path = "frame_save\\250.npy"
    # Extract the file name and add suffix
    save_name = base_name + "_setted"
    save_folder = "train_set"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)

    np.save(save_path, data_dict, allow_pickle=True)
    print(f"Data saved to {save_path}")
    ans = input("Finish? (y/n): ").strip().lower()
    if ans == 'y':
        return 0, 0
    else:
        return 1, 0


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib
import time

import cv2
import numpy as np


def extract_gray_features(image_gray_uint16, contour):
    """
    In a 16-bit or 8-bit grayscale image, extract two-dimensional features [mean, std] for the given contour.
    If you trained with different features, modify this function accordingly.
    """
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)

    # Create a mask
    h, w = image_gray_uint16.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, color=255, thickness=-1)

    if mask.sum() == 0:
        # If contour area is 0, return default features
        return [0.0, 0.0]

    mean_val, std_val = cv2.meanStdDev(image_gray_uint16, mask=mask)
    gray_mean = float(mean_val[0][0])
    gray_std = float(std_val[0][0])
    return [gray_mean, gray_std]


def build_dataset(train_set_dir):
    """
    Traverse all .npy files in 'train_set_dir',
    read data_dict, parse is_normal (label), contours, and original data (grayscale, uint16).
    Then extract features [grayscale mean, grayscale std] for each contour and store in X, y.
    """
    X = []
    y = []

    # 1. Loop through all .npy in the folder
    for filename in os.listdir(train_set_dir):
        if not filename.endswith(".npy"):
            continue  # Skip non-npy files

        npy_path = os.path.join(train_set_dir, filename)
        data_dict = np.load(npy_path, allow_pickle=True).item()

        # data_dict should contain "is_normal", "contours", "data"
        # data is a grayscale uint16 image, shape (H, W)
        image_gray_uint16 = data_dict["data"]
        contours = data_dict["contours"]
        is_normal_list = data_dict["is_normal"]  # list of True/False

        # 2. Extract features for each contour
        for cnt, normal_flag in zip(contours, is_normal_list):
            features = extract_gray_features(image_gray_uint16, cnt)
            X.append(features)

            # Convert True/False to 1/0 (or other binary labels)
            label = 1 if normal_flag else 0
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_model(train_set_dir):
    # ========== 1. Build the training dataset ==========
    X, y = build_dataset(train_set_dir)
    print("Feature matrix X shape:", X.shape)  # Expected (N, 2)
    print("Label y shape:", y.shape)           # Expected (N, )

    # If you're suspicious about the data range, you can print some features
    # print(X[:10], y[:10])

    # ========== 2. Split train/test set ==========
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ========== 3. Train RandomForest model ==========
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # ========== 4. Evaluate on the test set ==========
    y_pred = rf.predict(X_test)

    print("Random Forest classification report:")
    print(classification_report(y_test, y_pred, target_names=["Abnormal Rice", "Normal Rice"]))

    # ========== 5. (Optional) Save the model ==========
    model_save_path = "rice_rf_model.pkl"
    joblib.dump(rf, model_save_path)
    print(f"Model saved: {model_save_path}")


def classification(base_name,
                   model,  # Trained model object (not a path)
                   bg_gray,  # Background (16-bit grayscale array)
                   sample,   # Current frame (16-bit grayscale array)
                   min_area,
                   morph_kernel_size,
                   threshold_val
                   ):
    """
    For a single frame 'sample', perform foreground detection + RandomForest classification,
    then draw classification results (green or red boxes).
    Finally return (flag, result_img), where:
      - flag: 0 or 1, indicating whether the capture loop should continue (1=continue, 0=stop).
      - result_img: 8-bit BGR image containing recognition/classification results for real-time display.
    """
    # 1) Subtraction to get contours
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction(
        bg_gray=bg_gray,
        sample_gray=sample,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )
    # final_image is an 8-bit BGR image converted from sample_gray (already drawn with red contours)

    # 2) Extract features for each contour and predict
    pred_results = []
    for contour in valid_contours:
        features = extract_gray_features(sample, contour)
        # Reshape to (1,2)
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        pred_label = model.predict(features)[0]  # 0 / 1
        pred_results.append(pred_label)

    # 3) Redraw bounding boxes and text based on prediction results
    result_img = final_image.copy()
    for contour, pred_label in zip(valid_contours, pred_results):
        # Pred label: 0=Abnormal (red), 1=Normal (green)
        color = (0, 255, 0) if pred_label == 1 else (0, 0, 255)
        label_text = "Normal" if pred_label == 1 else "Abnormal"

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_img, label_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4) Optionally save the result image
    out_dir = "results_set"
    os.makedirs(out_dir, exist_ok=True)
    out_filename = base_name + "_inference.png"
    out_path = os.path.join(out_dir, out_filename)
    cv2.imwrite(out_path, result_img)

    # 5) Optionally save predicted labels for record
    pred_data = {
        "valid_contours": valid_contours,
        "pred_labels": pred_results
    }
    out_npy_path = os.path.join(out_dir, base_name + "_pred.npy")
    np.save(out_npy_path, pred_data, allow_pickle=True)

    # 6) If you need to trigger a stop under certain conditions, return (0, None) here
    #    For now, always return 1, indicating "continue."
    return 1, result_img


if __name__ == '__main__':

    exposure_time = 0.05
    frame_num = int(0.5 / exposure_time)

    # -----------------------
    # Step 1: Capture a background image
    # -----------------------
    answer_bg = input("Do you need to capture a background image? (y/n): ").strip().lower()
    if answer_bg == 'y':
        print("Starting to capture the background image...")
        capture_single_image(name='background', exposure_time=exposure_time)
        bg_gray_path = "background.npy"
    elif answer_bg == 'n':
        # If not needed, directly use an existing background, e.g. bg_gray=np.load("background.npy")
        bg_gray_path = "background.npy"
    else:
        bg_gray_path = "background.npy"

    bg_gray = np.load("background.npy")
    print(f"Background loaded, path used: {bg_gray_path}")

    # -----------------------
    # Step 2: Handling parameters
    # -----------------------
    answer_param = input("Do you need to tune detection parameters? (y/n): ").strip().lower()

    if answer_param == 'n':
        # Directly read from the local file detect_parameters.npy
        if not os.path.exists('detect_parameters.npy'):
            print("detect_parameters.npy does not exist. Cannot read parameters. Please tune first.")

    else:
        # Need to tune
        print("Starting parameter tuning process...")
        # First capture an image for tuning
        capture_single_image(name='find_parameters', exposure_time=exposure_time)  # Adjust exposure time as needed
        sample_path = 'find_parameters.npy'
        sample_gray = np.load(sample_path)

        # Use matplotlib to display
        while True:
            try:
                min_area = int(input("Please enter min_area (integer): "))
                morph_kernel_size = int(input("Please enter morph_kernel_size (integer): "))
                threshold_val = int(input("Please enter threshold_val (integer): "))
            except ValueError:
                print("Invalid input format, please try again.")
                continue

            final_img, fg_mask, valid_contours = detect_rice_by_subtraction(
                bg_gray=bg_gray,
                sample_gray=sample_gray,
                min_area=min_area,
                morph_kernel_size=morph_kernel_size,
                threshold_val=threshold_val
            )

            # Convert to RGB for proper matplotlib display
            show_final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

            # Display the result
            plt.figure(figsize=(8, 6))
            plt.imshow(show_final_img)
            plt.title(f"min_area={min_area}, morph_kernel_size={morph_kernel_size}, threshold_val={threshold_val}")
            plt.show()

            ans_satisfied = input("Are you satisfied with the current detection result? (y/n): ").strip().lower()
            if ans_satisfied == 'y':
                print("Okay, saving parameters to detect_parameters.npy.")
                params_dict = {
                    'min_area': min_area,
                    'morph_kernel_size': morph_kernel_size,
                    'threshold_val': threshold_val
                }
                np.save('detect_parameters.npy', params_dict)
                break
            else:
                print("Not satisfied, please re-enter parameters.")

    params_dict = np.load('detect_parameters.npy', allow_pickle=True).item()
    min_area = params_dict.get('min_area', 1000)
    morph_kernel_size = params_dict.get('morph_kernel_size', 2)
    threshold_val = params_dict.get('threshold_val', 4)
    print("Parameters read from detect_parameters.npy:")
    print(f"min_area = {min_area}, morph_kernel_size = {morph_kernel_size}, threshold_val = {threshold_val}")

    # -----------------------
    # Step 3: Set up the training set
    # -----------------------
    answer_st = input("Do you need to set up the training set? (y/n): ").strip().lower()
    if answer_st == 'y':
        capture_continuously(get_training_set, 0, bg_gray=bg_gray,
                             min_area=min_area,
                             morph_kernel_size=morph_kernel_size,
                             threshold_val=threshold_val,
                             exposure_time=exposure_time, frame_num=frame_num)
        train_set_dir = 'train_set'
    else:
        train_set_dir = 'train_set'
    print(f"Training set loaded, using path: {train_set_dir}")

    # -----------------------
    # Step 4: Train the model
    # -----------------------
    answer_st = input("Do you need to train the model? (y/n): ").strip().lower()
    if answer_st == 'y':
        train_model(train_set_dir)
        model_path = "rice_rf_model.pkl"
    else:
        model_path = "rice_rf_model.pkl"
    print(f"Model training complete, path: {model_path}")

    model = joblib.load(model_path)
    print("Model loaded.")

    capture_continuously(classification, model, bg_gray=bg_gray,
                         min_area=min_area,
                         morph_kernel_size=morph_kernel_size,
                         threshold_val=threshold_val,
                         exposure_time=exposure_time, frame_num=frame_num)
