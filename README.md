# MNIST CNN Classification Hobby Project  
**MNIST Digit Classification (PyTorch vs. TensorFlow/Keras)**

---

## Project Overview

This project targets beginners in Deep Learning who want a **complete, end-to-end understanding** of a machine learning workflow — from model definition and training to evaluation, deployment-style inference, and optimization.

A simple **Convolutional Neural Network (CNN)** is implemented to classify handwritten digits (0–9) using the **MNIST dataset**, in **both PyTorch and TensorFlow/Keras**.

The project is intentionally structured to highlight:
- Clean separation between training and inference
- Framework-specific design philosophies
- Educational deployment concepts (live camera inference)
- Model optimization readiness (quantization)

---

## Project Structure

The project enforces strict separation between frameworks to avoid conceptual and tooling confusion.

```
/mnist-cnn-quantization-project/
├── interactive_run.py                 # The main interactive script to start the project.
│
├── /pytorch/                   # PyTorch Implementation (Imperative/Research style)
│   ├── model_utils.py          # Defines the MNIST_CNN class (Architecture Blueprint).
│   ├── train.py                # Loads data, trains the model, saves weights (.pth).
│   └── test.py                 # Handles both Test Set Evaluation and Live Camera Feed.
│
└── /tensorflow_keras/          # TensorFlow/Keras Implementation (Declarative/Production style)
    ├── model_utils.py          # Defines the sequential model (Architecture Blueprint).
    ├── train.py                # Loads data, trains the model, saves weights (.keras).
    └── test.py                 # Handles both Test Set Evaluation and Live Camera Feed.
```

---

## Prerequisites

- Python 3.8 or higher
- pip package manager

### Install dependencies

```
pip install tensorflow keras torch torchvision numpy opencv-python
```

---
## How to Run the Project
This section guides you through setting up the environment, obtaining the code, and running the interactive application.

The entire project is controlled by the central `interactive_run.py` script. Although it is possible to run the programs separately for more in-depth comparison and study.

**Step 1: Clone the Repository**
```# Clone the repository and navigate into the project directory
git clone https://github.com/your-username/understanding-cnns.git
cd understanding-cnns
```

**Step 2: Run the Interactive Main Script**
```
python interactive_run.py
```

**Step 4: Follow the Interactive Menu**

The script guides you through the remaining steps in sequence:

* Framework Selection

  * Choose **PyTorch (1)** or **TensorFlow/Keras (2)**.

* Training

  * Automatically runs the selected framework’s `train.py`.
  * Trains the CNN.
  * Saves the trained model weights.

* Testing Mode Selection

  * Choose how the trained model is evaluated:

    * **1: Standard Test Set Evaluation**

      * Evaluates on the 10,000 unseen MNIST test images.
      * Reports final accuracy.
    * **2: Live Web Camera ROI**

      * Activates the webcam.
      * Captures a snapshot.
      * Allows ROI selection.
      * Runs the CNN on the selected region.

---

## CNNs for Beginners: The "Vision Machine" Guide
A Convolutional Neural Network (CNN) is a specialized type of deep learning model designed specifically to "look" at and understand images. Think of it as teaching a computer to see, not by giving it rules (like "a 7 has a horizontal line"), but by showing it thousands of examples.

### 1. Why Not a Standard Neural Network?
A standard neural network, i.e., a Multi-Layere Perceptron (MLP) sees an image only as a long, flat list of numbers (pixels). If you shift a digit by one pixel, the list of numbers changes completely, confusing the model.
A CNN fixes this by doing two smart things:
* Local Focus: It only looks at tiny parts of the image at a time.
* Feature Sharing: It looks for the same feature (like a diagonal edge) everywhere in the image.

### 2. The Feature Extractor Block (The "Looking" Part)
This block is where the model learns the visual vocabulary of the image.

**A. Convolution (Conv2D):** This is the core operation. It uses a tiny magnifying glass called a Kernel (or filter, usually 3×3 pixels).

* Action: The 3×3 kernel slides across the entire image. At each step, it calculates a weighted sum of the 9 pixels it covers.
* Result: The kernel doesn't see the whole image; it only learns to detect one specific feature, like a vertical line, a curve, or a corner. The result is a Feature Map showing where that specific feature exists in the image.
* Weights: The numbers inside the kernel are the weights the network learns during training.

**B. ReLU Activation:** After the convolution, the Rectified Linear Unit (ReLU) function is applied. This simply throws away all negative results (sets them to zero) and keeps the positive ones. This step introduces non-linearity, which is necessary for the network to model complex, curvy boundaries needed to distinguish shapes like '6' from '8'.

**C. Max-Pooling**
* Action: A Max-Pooling layer takes small non-overlapping windows (e.g., 2×2) from the feature map and keeps only the single highest number from that window.

* Result: This dramatically shrinks the size of the data (downsampling) and makes the features translation invariant. If a line moves slightly to the left, the max-pool value often remains the same, ensuring the classification stays robust.

### The Classification Head (The "Deciding" Part)
After several rounds of Convolution and Pooling, the image has been condensed into a powerful set of abstract features. This highly processed data is then handed off to a traditional network.

**A. Flatten:** The multi-dimensional feature map (e.g., 7×7×64) is simply stretched into a single, long 1D vector to prepare it for the final layers.

**B. Dense (FC) Layers:** These are the final, fully connected layers. They take the feature vector and combine all the learned information to make the final classification decision.

**Softmax - Ouput Layer:** The final layer has 10 neurons (for the digits 0-9) and uses the Softmax activation. This converts the raw numerical scores into probabilities that sum up to 1.0. The digit corresponding to the highest probability is the model's prediction.

### 4. The Learning Process
The entire process is governed by Backpropagation and the Optimizer (Adam), which constantly adjust the weights inside the 3×3 kernels to minimize the Loss Function (Cross-Entropy) until the model is accurate.

**Loss Function**
* Purpose: To measure how wrong the model's prediction is compared to the true answer. It quantifies the error.

* Action: For our MNIST project, we use Categorical Cross-Entropy. If the model predicts a '4' with 90% confidence, but the true label is a '9', the loss function returns a large number (a high penalty). If the prediction is accurate, the loss is close to zero.

* Goal: The entire training process is dedicated to minimizing this loss value.

**Backpropagation**
* Purpose: To efficiently calculate the gradient (the slope) of the loss function with respect to every single weight in the network.

* Action: The error calculated by the loss function travels backward from the output layer, through all the dense and convolutional layers, using the Chain Rule of Calculus.

* Result: This process tells the model exactly how much each weight contributed to the final error and which direction it needs to be adjusted.

**Optimizer**
Purpose: To use the gradients (calculated by Backpropagation) to update the model's weights in the direction that decreases the loss.

---

# Next Steps & Advanced Learning
This project serves as a perfect foundation for advanced topics:
1. Model Quantization (Optimization)
    Goal: Reduce the model size by converting weights from Float32 to Integer 8 (INT8).
    Files to Add: Create a quantize_deploy.py in both framework folders.
    	TensorFlow: Use the tf.lite.TFLiteConverter to convert the .keras model to an optimized .tflite file.
        PyTorch: Use the torch.quantization module with post-training static quantization.

2. Training Visualization
    Goal: See how the loss and accuracy change over time.
    Tool: Modify train.py to log history and use a library like Matplotlib to plot the training loss and validation loss curves.

3. Data Augmentation
    Goal: Improve generalization by artificially increasing the training data.
    Method: Introduce random small rotations or shifts to the MNIST images during training.
