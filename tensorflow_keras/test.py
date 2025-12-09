# test.py (TensorFlow/Keras) – snapshot → ROI → predict

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.keras")

# -----------------------------------------------------------------------------
# Camera → Snapshot → ROI → Predict
# -----------------------------------------------------------------------------

def run_camera_roi_inference(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nCamera started.")
    print("Press 's' to take snapshot, 'q' to quit.")

    snapshot = None

    # --- Live preview ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera Preview (press 's')", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            snapshot = frame.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if snapshot is None:
        return

    # --- ROI selection ---
    print("Draw bounding box around digit, press ENTER or SPACE to confirm.")
    x, y, w, h = cv2.selectROI(
        "Select Digit ROI", snapshot, fromCenter=False, showCrosshair=True
    )
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("Invalid ROI.")
        return

    digit_img = snapshot[y:y+h, x:x+w]

    # Preprocess ROI into MNIST format
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)
    digit = th[y:y+h, x:x+w]

    side = max(w, h)
    square = 255 * np.ones((side, side), dtype=np.uint8)
    square[(side-h)//2:(side-h)//2+h,
        (side-w)//2:(side-w)//2+w] = digit

    digit28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    model_input = digit28.astype("float32").reshape(1, 28, 28, 1) / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)

    # --- Inference ---
    prediction = model.predict(model_input, verbose=0)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # --- Display result ---
    cv2.putText(
        snapshot,
        f"Prediction: {predicted_digit} ({confidence*100:.1f}%)",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    cv2.imshow("Result", snapshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Standard test-set evaluation
# -----------------------------------------------------------------------------

def run_test_set_eval(model):
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("Evaluating on Test Set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"\nFinal Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")

    # Single prediction sanity check
    sample_image = x_test[0:1]
    prediction = model.predict(sample_image, verbose=0)
    print("\nExample Prediction on first test image:")
    print(f"Predicted: {np.argmax(prediction)}")
    print(f"True Label: {y_test[0]}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run train.py first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    if len(sys.argv) > 1 and sys.argv[1] == "camera_roi":
        run_camera_roi_inference(loaded_model)
    else:
        run_test_set_eval(loaded_model)