# interactive_run.py

import subprocess
import os
import sys

# --- Configuration ---
TF_DIR = 'tensorflow_keras'
PT_DIR = 'pytorch'
MODEL_FILENAME = 'mnist_cnn'


def run_script(directory, script_name, *args):
    """Executes a Python script in the specified directory."""
    script_path = os.path.join(directory, script_name)

    if not os.path.exists(script_path):
        print(f"\n[ERROR] Script not found: {script_path}")
        return False

    command = [sys.executable, script_path] + list(args)
    print(f"\n---> Executing: {' '.join(command)}")

    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy()  # propagate DISPLAY / QT / CUDA vars
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] Script execution failed in {directory}/{script_name}.")
        print(f"Details: {e}")
        return False


def interactive_menu():
    """Main orchestration logic."""

    # --- 1. Framework Selection ---
    while True:
        print("\n--- Framework Selection ---")
        print("1: PyTorch")
        print("2: TensorFlow/Keras")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == '1':
            chosen_dir = PT_DIR
            model_ext = ".pth"
            print("Selected: PyTorch.")
            break
        elif choice == '2':
            chosen_dir = TF_DIR
            model_ext = ".keras"
            print("Selected: TensorFlow/Keras.")
            break
        else:
            print("Invalid choice.")

    # --- 2. Check for Existing Model ---
    model_path = os.path.join(chosen_dir, f"{MODEL_FILENAME}{model_ext}")

    if os.path.exists(model_path):
        print(f"\n[INFO] Existing model found at: {model_path}")
        print("Skipping training.")
    else:
        print("\n--- 🏋️ Starting Model Training ---")
        if not run_script(chosen_dir, 'train.py'):
            print("\nModel training failed. Aborting.")
            return

    # --- 3. Testing Mode Selection ---
    while True:
        print("\n--- 🔎 Testing Mode Selection ---")
        print("1: Standard Test Set Evaluation")
        print("2: Capture → Select ROI → Predict")
        mode = input("Enter choice (1 or 2): ").strip()

        if mode == '1':
            test_mode_arg = "test_set"
            print("Selected: Standard Test Set Evaluation.")
            break
        elif mode == '2':
            test_mode_arg = "camera_roi"
            print("Selected: Capture → Select ROI → Predict.")
            break
        else:
            print("Invalid choice.")

    # --- 4. Execute Testing ---
    print("\n--- Starting Model Evaluation ---")
    run_script(chosen_dir, 'test.py', test_mode_arg)

    print("\nExecuted")


if __name__ == "__main__":
    interactive_menu()
