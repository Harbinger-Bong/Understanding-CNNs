# test.py (PyTorch) – updated with snapshot → ROI → predict flow
import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_utils import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.pth")

# 0. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_camera_roi_inference(model, device):
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

    # 1. Grayscale
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

    # 2. Blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adaptive threshold
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 4. Extract digit bounding box
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)
    digit = th[y:y+h, x:x+w]

    # 5. Make square canvas
    side = max(w, h)
    square = 255 * np.ones((side, side), dtype=np.uint8)
    square[(side-h)//2:(side-h)//2+h,
        (side-w)//2:(side-w)//2+w] = digit

    # 6. Resize to 28×28
    digit28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # 7. Normalize
    input_img = digit28.astype("float32") / 255.0

    # 8. Convert to tensor (PyTorch format)
    tensor_img = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).to(device)

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        output = model(tensor_img)
        probabilities = F.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted].item()

    # --- Display result ---
    cv2.putText(
        snapshot,
        f"Prediction: {predicted} ({confidence*100:.1f}%)",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    cv2.imshow("Result", snapshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_test_set_eval(model):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root=os.path.join(BASE_DIR, "data"),
        train=False,
        download=False,
        transform=transform
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    print("Evaluating on Test Set...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # 1. Instantiate model
    model = create_model().to(device)

    # 2. Load weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run train.py first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    # 3. Mode selection
    if len(sys.argv) > 1 and sys.argv[1] == "camera_roi":
        run_camera_roi_inference(model, device)
    else:
        run_test_set_eval(model)