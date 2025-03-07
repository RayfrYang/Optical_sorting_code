import os             # For handling file paths and directories
import time           # For timing and FPS calculations
import numpy as np    # For numerical computations
import cv2            # For image processing
import joblib         # For saving/loading trained models
import matplotlib.pyplot as plt  # For image visualization

# Attempt to import pylablib (Thorlabs devices)
try:
    import pylablib as pll
    from pylablib.devices import Thorlabs
except ImportError:
    print("[WARN] Unable to import pylablib / Thorlabs SDK.")

def capture_single_image_color(name='color_background', exposure_time=0.04):
    """
    Capture a single color image using a Thorlabs camera and save it as a .npy file.
    """
    # Initialize the Thorlabs camera (make sure it's connected/configured).
    cam = Thorlabs.ThorlabsTLCamera()

    # Set exposure time (in seconds).
    cam.set_exposure(exposure_time)

    # Prepare to acquire frames (specify how many frames).
    cam.setup_acquisition(nframes=1)
    cam.start_acquisition()
    try:
        # Wait for a single frame, then read it.
        cam.wait_for_frame(timeout=10.0)
        frame_color = cam.read_newest_image()

        # Print basic info about the captured image (shape, data type).
        print("Shape:", frame_color.shape, "Dtype:", frame_color.dtype)

        # Save the image to disk as a NumPy array.
        np.save(name + '.npy', frame_color)
        print(f"[INFO] Saved color image as {name}.npy")
    finally:
        # Always close the camera (even if errors occur).
        cam.close()
        print("[INFO] Camera closed.")

def capture_continuously_color(
    func,
    model,
    bg_color=None,
    min_area=500,
    morph_kernel_size=3,
    threshold_val=30,
    exposure_time=0.05,
    frame_num=10
):
    """
    Continuously capture color images, call a custom processing function (func)
    on each frame, and display the result in real-time. The function 'func' must
    accept parameters (file_index, model, bg_color, frame, min_area, morph_kernel_size, threshold_val)
    and return (flag, result_img).
    """
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(exposure_time)
    # Prepare to acquire multiple frames (frame_num frames per iteration).
    cam.setup_acquisition(nframes=frame_num)
    cam.start_acquisition()

    frame_count = 0     # Count how many frames we have processed in the last second
    last_time = time.time()
    last_frame_time = last_time
    file_index = 0      # Used to name or track frames

    print(f"[INFO] Continuous capture started, target FPS: {frame_num}.")
    try:
        while True:
            now = time.time()
            # Ensure we capture frames at the desired frame rate
            if (now - last_frame_time) >= 1.0 / frame_num:
                frame_color = cam.read_newest_image()
                if frame_color is None:
                    continue

                # Process the latest frame with the user-provided function
                flag, result_img = func(
                    str(file_index),
                    model,
                    bg_color,
                    frame_color,
                    min_area,
                    morph_kernel_size,
                    threshold_val
                )

                # If the function signals "stop" (flag == 0), break the loop
                if flag == 0:
                    print("[INFO] Function returned 0, stopping.")
                    break

                # Display the processed frame in a window
                cv2.imshow("Live Preview (Color)", result_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] ESC pressed, exiting.")
                    break

                file_index += 1
                frame_count += 1
                last_frame_time = now

            # Once a second, print the actual measured FPS
            if (now - last_time) >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"[INFO] Actual FPS: {fps:.2f}")
                frame_count = 0
                last_time = now
    except KeyboardInterrupt:
        print("[INFO] Stopped by KeyboardInterrupt.")
    finally:
        # Clean up: close camera, close windows
        cam.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed, end.")

def detect_rice_by_subtraction_color(bg_color, sample_color, min_area=500, morph_kernel_size=3, threshold_val=30):
    """
    Simple background subtraction for color images:
    1) Absolute difference
    2) Convert to grayscale
    3) Threshold and morphological operations
    4) Find valid contours and draw them in red
    Returns: (final_image, fg_mask, valid_contours)
    """
    # Calculate absolute difference between background and current frame
    diff_bgr = cv2.absdiff(bg_color, sample_color)

    # Convert the difference to grayscale
    diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image -> binary mask
    _, fg_mask = cv2.threshold(diff_gray, threshold_val, 255, cv2.THRESH_BINARY)

    # Morphological operations (open -> remove small noise, close -> fill small holes)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find connected contours in the binary mask
    contours, _ = cv2.findContours(fg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the original frame for drawing
    final_image = sample_color.copy().astype(np.uint8)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter out small contours by area threshold
        if area < min_area:
            continue
        valid_contours.append(cnt)
        # Draw contour in red
        cv2.drawContours(final_image, [cnt], -1, (0, 0, 255), 2)

    return final_image, fg_mask, valid_contours

def label_contours(image, contours):
    """
    Label each contour with an index for later reference.
    """
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        # Put text (contour index) at the centroid
        cv2.putText(image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image

def get_training_set_color(base_name, unused_model, bg_color, sample_color, min_area, morph_kernel_size, threshold_val):
    """
    Perform background subtraction on a color image, label each contour, and ask the user
    whether it is 'normal' or not. Saves labeled data to a .npy file for training.
    - base_name: used in the output file name
    - unused_model: placeholder, not used here
    - bg_color, sample_color: 16-bit BGR images
    - returns: (flag, None) so that capture_continuously_color can decide whether to stop
    """
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction_color(
        bg_color=bg_color,
        sample_color=sample_color,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )
    labeled_img = label_contours(final_image, valid_contours)

    # Create a window for labeling
    cv2.namedWindow('Labeling Window (Color)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Labeling Window (Color)', 800, 600)
    cv2.imshow('Labeling Window (Color)', labeled_img)

    normal_flags = []
    filtered_contours = []

    # Ask the user for each contour: is it normal?
    for i, cnt in enumerate(valid_contours):
        cv2.waitKey(1)
        user_input = input(f"Contour {i}, is it normal? (y/n/e): ").strip().lower()
        if user_input in ["y", "yes"]:
            normal_flags.append(True)
            filtered_contours.append(cnt)
        elif user_input in ["n", "no"]:
            normal_flags.append(False)
            filtered_contours.append(cnt)
        else:
            print(f"Contour {i} ignored.")

    cv2.destroyAllWindows()

    # Convert contours to arrays for easier saving
    conts_as_arrays = [c.reshape(-1, 2) for c in filtered_contours]
    data_dict = {
        "is_normal": normal_flags,
        "contours": conts_as_arrays,
        "data": sample_color
    }

    # Save to train_set folder
    os.makedirs("train_set", exist_ok=True)
    save_path = os.path.join("train_set", base_name + "_setted.npy")
    np.save(save_path, data_dict, allow_pickle=True)
    print(f"[INFO] Saved labeled data to: {save_path}")

    # Ask user if we should stop labeling
    ans = input("Finish labeling? (y=finish/n=continue): ").strip().lower()
    if ans == "y":
        return 0, 0
    else:
        return 1, 0

def extract_color_features_16bit(image_bgr_uint16, contour):
    """
    Extract color features from the region defined by the contour (16-bit BGR image).
    Returns [R_mean, G_mean, B_mean, R_std, G_std, B_std].
    """
    # Ensure the contour is in the correct shape
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)

    # Compute bounding rect of the contour
    x, y, w, h = cv2.boundingRect(contour)
    roi = image_bgr_uint16[y:y+h, x:x+w]

    # Shift the contour relative to the local region
    contour_shifted = contour.copy()
    contour_shifted[..., 0] -= x
    contour_shifted[..., 1] -= y

    # Create a local mask corresponding to the contour
    mask_local = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_local, [contour_shifted.astype(np.int32)], -1, 255, -1)

    # Calculate mean and std dev of B, G, R inside the contour
    mean_val, std_val = cv2.meanStdDev(roi, mask=mask_local)
    if mask_local.sum() == 0:
        return [0,0,0,0,0,0]
    B_mean, G_mean, R_mean = mean_val.flatten()
    B_std, G_std, R_std = std_val.flatten()
    # Return the features in [R_mean, G_mean, B_mean, R_std, G_std, B_std] format
    return [R_mean, G_mean, B_mean, R_std, G_std, B_std]

def build_dataset(train_set_dir):
    """
    Traverse saved training data (*.npy) in train_set_dir,
    extract features, and form a dataset (X, y).
    """
    X, y = [], []
    for fname in os.listdir(train_set_dir):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(train_set_dir, fname)
        data_dict = np.load(path, allow_pickle=True).item()

        # The original 16-bit color image
        image_bgr_uint16 = data_dict["data"]
        contours = data_dict["contours"]
        normal_flags = data_dict["is_normal"]

        # Extract features from each contour
        for cnt, flag in zip(contours, normal_flags):
            feat = extract_color_features_16bit(image_bgr_uint16, cnt)
            X.append(feat)
            y.append(1 if flag else 0)

    # Convert to NumPy arrays
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_model(train_set_dir):
    """
    Build a dataset from labeled files, train a RandomForest model,
    and save the model to 'rice_rf_model.pkl'.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Build the dataset from labeled .npy files
    X, y = build_dataset(train_set_dir)
    print("X shape:", X.shape, "y shape:", y.shape)

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Abnormal","Normal"]))

    # Save the trained model
    joblib.dump(rf, "rice_rf_model.pkl")
    print("[INFO] Model saved as rice_rf_model.pkl")

def classification_color(base_name, model, bg_color, sample_color, min_area, morph_kernel_size, threshold_val):
    """
    Perform background subtraction on a color frame, extract features, classify each contour,
    and draw bounding boxes with labels on the result.
    Returns (1, annotated_image).
    """
    # Detect possible "rice" or objects based on background subtraction
    final_img, fg_mask, valid_contours = detect_rice_by_subtraction_color(
        bg_color=bg_color,
        sample_color=sample_color,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )

    # Predict "normal" (1) or "abnormal" (0) for each contour
    pred_results = []
    for cnt in valid_contours:
        feats = extract_color_features_16bit(sample_color, cnt)
        feats = np.array(feats, dtype=np.float32).reshape(1, -1)
        label = model.predict(feats)[0]
        pred_results.append(label)

    # Draw bounding boxes and labels on the final image
    result_img = final_img.copy()
    for cnt, label in zip(valid_contours, pred_results):
        color = (0,255,0) if label == 1 else (0,0,255)
        text = "Normal" if label == 1 else "Abnormal"
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(result_img, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the result image for reference
    out_dir = "results_inference"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, base_name + "_color_infer.png")
    cv2.imwrite(out_png, result_img)
    return 1, result_img

if __name__ == '__main__':
    # Ask whether to capture a new background image
    ans_bg = input("Take a COLOR background image? (y/n): ").strip().lower()
    if ans_bg == "y":
        capture_single_image_color(name="color_background", exposure_time=0.02)
        bg_color_path = "color_background.npy"
    else:
        bg_color_path = "color_background.npy"

    # Make sure the background file exists before proceeding
    if not os.path.exists(bg_color_path):
        raise FileNotFoundError("Background image not found. Please capture or specify an existing file.")
    bg_color = np.load(bg_color_path)
    print(f"[INFO] Loaded color background: {bg_color_path}")

    # Ask whether to tune detection parameters
    ans_param = input("Tune detection parameters? (y/n): ").strip().lower()
    if ans_param == "y":
        capture_single_image_color(name="find_parameters_color", exposure_time=0.02)
        sample_path = "find_parameters_color.npy"
        sample_color = np.load(sample_path)

        while True:
            try:
                min_area = int(input("Enter min_area: "))
                morph_kernel_size = int(input("Enter morph_kernel_size: "))
                threshold_val = int(input("Enter threshold_val: "))
            except ValueError:
                print("[WARN] Invalid input format.")
                continue

            final_img, fg_mask, valid_contours = detect_rice_by_subtraction_color(
                bg_color=bg_color,
                sample_color=sample_color,
                min_area=min_area,
                morph_kernel_size=morph_kernel_size,
                threshold_val=threshold_val
            )

            # Convert BGR to RGB for matplotlib display
            show_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(show_img)
            plt.title(f"min_area={min_area}, morph={morph_kernel_size}, thresh={threshold_val}")
            plt.show()

            ans_ok = input("Satisfied with these parameters? (y/n): ").strip().lower()
            if ans_ok == "y":
                print("Saving to detect_parameters_color.npy")
                params_dict = {
                    "min_area": min_area,
                    "morph_kernel_size": morph_kernel_size,
                    "threshold_val": threshold_val
                }
                np.save("detect_parameters_color.npy", params_dict)
                break
    else:
        # Use existing detection_parameters file or default
        if not os.path.exists("detect_parameters_color.npy"):
            print("[WARN] detect_parameters_color.npy not found, using default.")
            min_area = 500
            morph_kernel_size = 3
            threshold_val = 30
        else:
            params = np.load("detect_parameters_color.npy", allow_pickle=True).item()
            min_area = params.get("min_area", 500)
            morph_kernel_size = params.get("morph_kernel_size", 3)
            threshold_val = params.get("threshold_val", 30)

    print(f"[INFO] Using parameters: min_area={min_area}, morph_kernel_size={morph_kernel_size}, threshold_val={threshold_val}")

    # Ask whether to set up (collect) a new training set
    ans_ts = input("Set up training set? (y/n): ").strip().lower()
    if ans_ts == "y":
        exposure_time = 0.02
        # Example: how many frames to capture per second
        frame_num = int(0.5 / exposure_time)
        capture_continuously_color(
            get_training_set_color,
            model=0,
            bg_color=bg_color,
            min_area=min_area,
            morph_kernel_size=morph_kernel_size,
            threshold_val=threshold_val,
            exposure_time=exposure_time,
            frame_num=frame_num
        )

    # Ask whether to train the model based on collected training set
    ans_tr = input("Train model? (y/n): ").strip().lower()
    if ans_tr == "y":
        train_model(train_set_dir="train_set")

    # Attempt to load the model file
    model_path = "rice_rf_model.pkl"
    if not os.path.exists(model_path):
        print("[WARN] Model file not found. Classification is not possible without a model.")
    else:
        ans_run = input("Start continuous COLOR classification? (y/n): ").strip().lower()
        if ans_run == "y":
            model = joblib.load(model_path)
            print(f"[INFO] Model loaded: {model_path}")
            exposure_time = 0.02
            frame_num = int(0.5 / exposure_time)
            capture_continuously_color(
                classification_color,
                model=model,
                bg_color=bg_color,
                min_area=min_area,
                morph_kernel_size=morph_kernel_size,
                threshold_val=threshold_val,
                exposure_time=exposure_time,
                frame_num=frame_num
            )

    print("[INFO] Process completed.")
