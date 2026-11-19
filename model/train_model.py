"""
Model Training and Optimization Pipeline
AI-Based Crop Health Monitoring and Early Disease Detection System
Performs actual CNN training with cross-entropy loss, backpropagation & validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import os
import json
import time
from config import CLASSES, IMAGE_SIZE, CNN_WEIGHTS_PATH, MODEL_DIR
from model.cnn_model import CropHealthCNN
from model.dataset_generator import generate_image_for_class, generate_sample_images

def create_training_batch(samples_per_class=12, augment=True):
    """
    Creates an agricultural training set with data augmentation:
    - Random rotation (90, 180, 270)
    - Horizontal & vertical flips
    - Subtle brightness jitter
    """
    X_list = []
    y_list = []

    for cls_idx, cls_name in enumerate(CLASSES):
        for s in range(samples_per_class):
            seed = 1000 + cls_idx * 100 + s
            img_bgr = generate_image_for_class(cls_name, random_seed=seed)
            
            # Preprocess to 128x128 RGB normalized
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

            if augment and s > 0:
                # Random flip
                if np.random.rand() > 0.5:
                    img_resized = cv2.flip(img_resized, 1)
                # Random brightness
                factor = np.random.uniform(0.85, 1.15)
                img_resized = np.clip(img_resized.astype(np.float32) * factor, 0, 255).astype(np.uint8)

            img_norm = img_resized.astype(np.float32) / 255.0
            X_list.append(img_norm)
            y_list.append(cls_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y

def train_cnn(epochs=20, learning_rate=0.03):
    """
    Trains the CropHealthCNN model using categorical cross-entropy and backpropagation.
    Extracts multi-scale convolutional visual features and optimizes the classification head.
    """
    print("=" * 60)
    print("AI-Based Crop Health Monitoring - CNN Model Training")
    print("=" * 60)

    # First ensure sample images exist
    generate_sample_images()

    print("Generating augmented training dataset across 10 agricultural classes...")
    X_train, y_train = create_training_batch(samples_per_class=10, augment=True)
    X_val, y_val = create_training_batch(samples_per_class=3, augment=False)
    
    num_samples = len(X_train)
    num_classes = len(CLASSES)
    print(f"Training samples: {num_samples} | Validation samples: {len(X_val)}")
    print(f"Input dimensions: {X_train.shape[1:]} | Target classes: {num_classes}")

    # One-hot encode targets
    Y_train_onehot = np.zeros((num_samples, num_classes), dtype=np.float32)
    Y_train_onehot[np.arange(num_samples), y_train] = 1.0

    cnn = CropHealthCNN(num_classes=num_classes)

    print("\nExtracting CNN representations (Conv1 -> MaxPool -> Conv2 -> MaxPool -> Conv3 -> GAP)...")
    t0 = time.time()
    
    # Extract features in mini-batches
    train_features = []
    for i in range(0, num_samples, 20):
        batch = X_train[i:i+20]
        _, acts = cnn.forward(batch)
        train_features.append(acts["features"])
    F_train = np.vstack(train_features) # (num_samples, 64)

    val_features = []
    for i in range(0, len(X_val), 20):
        batch = X_val[i:i+20]
        _, acts = cnn.forward(batch)
        val_features.append(acts["features"])
    F_val = np.vstack(val_features)     # (len(X_val), 64)

    print(f"Feature extraction complete in {time.time() - t0:.2f}s. Shape: {F_train.shape}")
    print("\nTraining CNN Classification Head via Backpropagation & Cross-Entropy...")

    # Adam Optimizer parameters
    m_W_d1 = np.zeros_like(cnn.W_d1)
    v_W_d1 = np.zeros_like(cnn.W_d1)
    m_b_d1 = np.zeros_like(cnn.b_d1)
    v_b_d1 = np.zeros_like(cnn.b_d1)

    m_W_d2 = np.zeros_like(cnn.W_d2)
    v_W_d2 = np.zeros_like(cnn.W_d2)
    m_b_d2 = np.zeros_like(cnn.b_d2)
    v_b_d2 = np.zeros_like(cnn.b_d2)

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    step = 0

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(num_samples)
        F_shuff = F_train[perm]
        Y_shuff = Y_train_onehot[perm]
        y_shuff = y_train[perm]

        batch_size = 20
        epoch_loss = 0.0
        correct = 0

        for b in range(0, num_samples, batch_size):
            step += 1
            fb = F_shuff[b:b+batch_size]
            yb = Y_shuff[b:b+batch_size]
            yl = y_shuff[b:b+batch_size]
            m = len(fb)

            # Forward
            d1 = np.dot(fb, cnn.W_d1) + cnn.b_d1
            a1 = cnn.relu(d1)
            logits = np.dot(a1, cnn.W_d2) + cnn.b_d2
            probs = cnn.softmax(logits)

            # Loss
            loss = -np.sum(yb * np.log(np.clip(probs, 1e-7, 1.0 - 1e-7))) / m
            epoch_loss += loss * m

            # Accuracy
            preds = np.argmax(probs, axis=1)
            correct += np.sum(preds == yl)

            # Gradients
            d_logits = (probs - yb) / m
            dW2 = np.dot(a1.T, d_logits) + 0.0005 * cnn.W_d2
            db2 = np.sum(d_logits, axis=0)

            da1 = np.dot(d_logits, cnn.W_d2.T)
            dz1 = da1 * (d1 > 0)
            dW1 = np.dot(fb.T, dz1) + 0.0005 * cnn.W_d1
            db1 = np.sum(dz1, axis=0)

            # Adam updates for W2, b2
            m_W_d2 = beta1 * m_W_d2 + (1 - beta1) * dW2
            v_W_d2 = beta2 * v_W_d2 + (1 - beta2) * (dW2 ** 2)
            m_W_d2_hat = m_W_d2 / (1 - beta1 ** step)
            v_W_d2_hat = v_W_d2 / (1 - beta2 ** step)
            cnn.W_d2 -= learning_rate * m_W_d2_hat / (np.sqrt(v_W_d2_hat) + eps)

            m_b_d2 = beta1 * m_b_d2 + (1 - beta1) * db2
            v_b_d2 = beta2 * v_b_d2 + (1 - beta2) * (db2 ** 2)
            m_b_d2_hat = m_b_d2 / (1 - beta1 ** step)
            v_b_d2_hat = v_b_d2 / (1 - beta2 ** step)
            cnn.b_d2 -= learning_rate * m_b_d2_hat / (np.sqrt(v_b_d2_hat) + eps)

            # Adam updates for W1, b1
            m_W_d1 = beta1 * m_W_d1 + (1 - beta1) * dW1
            v_W_d1 = beta2 * v_W_d1 + (1 - beta2) * (dW1 ** 2)
            m_W_d1_hat = m_W_d1 / (1 - beta1 ** step)
            v_W_d1_hat = v_W_d1 / (1 - beta2 ** step)
            cnn.W_d1 -= learning_rate * m_W_d1_hat / (np.sqrt(v_W_d1_hat) + eps)

            m_b_d1 = beta1 * m_b_d1 + (1 - beta1) * db1
            v_b_d1 = beta2 * v_b_d1 + (1 - beta2) * (db1 ** 2)
            m_b_d1_hat = m_b_d1 / (1 - beta1 ** step)
            v_b_d1_hat = v_b_d1 / (1 - beta2 ** step)
            cnn.b_d1 -= learning_rate * m_b_d1_hat / (np.sqrt(v_b_d1_hat) + eps)

        train_acc = (correct / num_samples) * 100.0
        avg_loss = epoch_loss / num_samples

        if epoch % 5 == 0 or epoch == epochs:
            # Val accuracy
            d1_v = cnn.relu(np.dot(F_val, cnn.W_d1) + cnn.b_d1)
            probs_v = cnn.softmax(np.dot(d1_v, cnn.W_d2) + cnn.b_d2)
            val_preds = np.argmax(probs_v, axis=1)
            val_acc = (np.sum(val_preds == y_val) / len(y_val)) * 100.0
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    # Save weights
    cnn.save_weights(CNN_WEIGHTS_PATH)
    print(f"\nTrained CNN weights successfully saved to: {CNN_WEIGHTS_PATH}")
    return cnn

if __name__ == "__main__":
    # Remove older weights if present for fresh training
    if os.path.exists(CNN_WEIGHTS_PATH):
        os.remove(CNN_WEIGHTS_PATH)
    train_cnn(epochs=40, learning_rate=0.015)
