# train.py
import tensorflow as tf
from model_utils import create_model # Import our blueprint
import os

# 1. Load Data (Only Training Data needed here, but load_data returns both)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocessing
# Reshape and Normalize
x_train = x_train.astype('float32').reshape(-1, 28, 28, 1) / 255.0

# 3. Create Model
print("Building model...")
model = create_model()

# 4. Train
print("Starting training...")
history = model.fit(x_train, y_train, 
                    epochs=5,      # kept small for quick testing
                    batch_size=64, 
                    validation_split=0.1)

# 5. Save the Model
# We save it to a specific format (.keras is the modern standard)
save_path = 'mnist_cnn.keras'
model.save(save_path)
print(f"Model saved successfully to {save_path}")