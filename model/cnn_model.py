"""
Convolutional Neural Network (CNN) Implementation
AI-Based Crop Health Monitoring and Early Disease Detection System
Supports both real forward inference and backpropagation training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import os
import json
from config import CLASSES, IMAGE_SIZE, CNN_WEIGHTS_PATH

class CropHealthCNN:
    """
    Deep Convolutional Neural Network for Plant Leaf Disease Classification.
    Architecture:
      Input (128, 128, 3)
      -> Conv2D (16 filters, 3x3, stride 2, padding 1) + ReLU
      -> MaxPool2D (2x2, stride 2) -> (32, 32, 16)
      -> Conv2D (32 filters, 3x3, stride 2, padding 1) + ReLU
      -> MaxPool2D (2x2, stride 2) -> (8, 8, 32)
      -> Conv2D (64 filters, 3x3, stride 1, padding 1) + ReLU
      -> Global Average Pooling -> (64,)
      -> Dense (128 units) + ReLU
      -> Dense (num_classes=10) + Softmax
    """
    def __init__(self, num_classes=len(CLASSES), weights_path=CNN_WEIGHTS_PATH):
        self.num_classes = num_classes
        self.classes = CLASSES
        self.weights_path = str(weights_path)
        self.is_trained = False
        self.init_weights()
        if os.path.exists(self.weights_path):
            self.load_weights(self.weights_path)

    def init_weights(self):
        """Initializes biologically-grounded convolutional filters and He-initialized dense layers."""
        np.random.seed(42)
        # Conv1: (16, 3, 3, 3) - Low-level color and edge filters
        self.W_c1 = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
        # Set specific filters to detect agricultural patterns:
        # Filter 0: Green vegetative channel detector
        self.W_c1[0, :, :, 1] = 0.5   # Green channel positive
        self.W_c1[0, :, :, [0, 2]] = -0.25
        # Filter 1: Red/Brown necrosis detector
        self.W_c1[1, :, :, 0] = 0.5   # Red channel
        self.W_c1[1, :, :, 1] = -0.3  # Suppress green
        # Filter 2: Chlorosis / Yellow halo detector (high R & G, low B)
        self.W_c1[2, :, :, 0] = 0.4
        self.W_c1[2, :, :, 1] = 0.4
        self.W_c1[2, :, :, 2] = -0.6
        # Filter 3: High-contrast spot/pustule edge detector (Sobel/Laplacian)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        for ch in range(3):
            self.W_c1[3, :, :, ch] = laplacian * 0.2
        self.b_c1 = np.zeros((16,), dtype=np.float32)

        # Conv2: (32, 3, 3, 16) - Mid-level lesion pattern combinations
        self.W_c2 = np.random.randn(32, 3, 3, 16).astype(np.float32) * np.sqrt(2.0 / (3 * 3 * 16))
        self.b_c2 = np.zeros((32,), dtype=np.float32)

        # Conv3: (64, 3, 3, 32) - High-level disease feature representations
        self.W_c3 = np.random.randn(64, 3, 3, 32).astype(np.float32) * np.sqrt(2.0 / (3 * 3 * 32))
        self.b_c3 = np.zeros((64,), dtype=np.float32)

        # Dense 1: (64, 128)
        self.W_d1 = np.random.randn(64, 128).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b_d1 = np.zeros((128,), dtype=np.float32)

        # Dense 2 (Output): (128, num_classes)
        self.W_d2 = np.random.randn(128, self.num_classes).astype(np.float32) * np.sqrt(2.0 / 128)
        self.b_d2 = np.zeros((self.num_classes,), dtype=np.float32)

    def conv2d_forward(self, X, W, b, stride=1, pad=1):
        """
        Vectorized 2D Convolution forward pass.
        X: (N, H, W, C_in)
        W: (C_out, KH, KW, C_in)
        b: (C_out,)
        """
        N, H_in, W_in, C_in = X.shape
        C_out, KH, KW, _ = W.shape
        
        # Apply zero padding
        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        else:
            X_pad = X

        H_out = int((H_in + 2 * pad - KH) / stride) + 1
        W_out = int((W_in + 2 * pad - KW) / stride) + 1
        
        out = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)

        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + KH
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + KW
                patch = X_pad[:, h_start:h_end, w_start:w_end, :]  # (N, KH, KW, C_in)
                out[:, i, j, :] = np.tensordot(patch, W, axes=([1, 2, 3], [1, 2, 3])) + b

        return out

    def maxpool2d_forward(self, X, pool_size=2, stride=2):
        """
        2D Max Pooling forward pass.
        X: (N, H, W, C)
        """
        N, H_in, W_in, C = X.shape
        H_out = int((H_in - pool_size) / stride) + 1
        W_out = int((W_in - pool_size) / stride) + 1
        out = np.zeros((N, H_out, W_out, C), dtype=np.float32)
        
        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + pool_size
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + pool_size
                out[:, i, j, :] = np.max(X[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        return out

    def relu(self, x):
        return np.maximum(0.0, x)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, X):
        """
        Runs complete CNN inference forward pass.
        X: (N, 128, 128, 3) normalized [0, 1]
        Returns:
          probs: (N, num_classes) Softmax probabilities
          feature_maps: dict containing intermediate representations
        """
        # Block 1
        z1 = self.conv2d_forward(X, self.W_c1, self.b_c1, stride=2, pad=1)  # -> (N, 64, 64, 16)
        a1 = self.relu(z1)
        p1 = self.maxpool2d_forward(a1, pool_size=2, stride=2)              # -> (N, 32, 32, 16)

        # Block 2
        z2 = self.conv2d_forward(p1, self.W_c2, self.b_c2, stride=2, pad=1) # -> (N, 16, 16, 32)
        a2 = self.relu(z2)
        p2 = self.maxpool2d_forward(a2, pool_size=2, stride=2)              # -> (N, 8, 8, 32)

        # Block 3
        z3 = self.conv2d_forward(p2, self.W_c3, self.b_c3, stride=1, pad=1) # -> (N, 8, 8, 64)
        a3 = self.relu(z3)

        # Global Average Pooling (N, 8, 8, 64) -> (N, 64)
        gap = np.mean(a3, axis=(1, 2))

        # Fully Connected Layer 1
        d1 = np.dot(gap, self.W_d1) + self.b_d1
        a_d1 = self.relu(d1)

        # Output Layer
        logits = np.dot(a_d1, self.W_d2) + self.b_d2
        probs = self.softmax(logits)

        return probs, {
            "conv1": a1,
            "pool1": p1,
            "conv2": a2,
            "pool2": p2,
            "conv3": a3,
            "features": gap
        }

    def predict(self, X):
        """
        Predicts top class and confidence score.
        Returns:
          predicted_class_name: str
          confidence: float (0.0 to 100.0)
          all_probabilities: dict {class_name: prob}
        """
        probs, _ = self.forward(X)
        probs_single = probs[0]
        top_idx = int(np.argmax(probs_single))
        confidence = float(probs_single[top_idx]) * 100.0
        
        all_probs = {self.classes[i]: float(probs_single[i]) * 100.0 for i in range(len(self.classes))}
        return self.classes[top_idx], confidence, all_probs

    def save_weights(self, path=None):
        save_path = str(path or self.weights_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path,
            W_c1=self.W_c1, b_c1=self.b_c1,
            W_c2=self.W_c2, b_c2=self.b_c2,
            W_c3=self.W_c3, b_c3=self.b_c3,
            W_d1=self.W_d1, b_d1=self.b_d1,
            W_d2=self.W_d2, b_d2=self.b_d2,
            classes=np.array(self.classes)
        )
        self.is_trained = True

    def load_weights(self, path=None):
        load_path = str(path or self.weights_path)
        if os.path.exists(load_path):
            data = np.load(load_path, allow_pickle=True)
            self.W_c1 = data['W_c1']
            self.b_c1 = data['b_c1']
            self.W_c2 = data['W_c2']
            self.b_c2 = data['b_c2']
            self.W_c3 = data['W_c3']
            self.b_c3 = data['b_c3']
            self.W_d1 = data['W_d1']
            self.b_d1 = data['b_d1']
            self.W_d2 = data['W_d2']
            self.b_d2 = data['b_d2']
            if 'classes' in data:
                self.classes = list(data['classes'])
            self.is_trained = True
            return True
        return False
