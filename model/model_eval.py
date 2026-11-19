"""
Model Evaluation and Metrics Module
AI-Based Crop Health Monitoring and Early Disease Detection System
Computes Accuracy, Precision, Recall, F1-Score and Confusion Matrix on held-out test data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import json
import os
from config import CLASSES, IMAGE_SIZE, CNN_WEIGHTS_PATH, MODEL_DIR
from model.cnn_model import CropHealthCNN
from model.dataset_generator import generate_image_for_class

def evaluate_model(test_samples_per_class=10):
    """
    Evaluates the trained CNN on a completely unseen held-out test split.
    Calculates:
      - Overall Accuracy
      - Per-class Precision, Recall, F1-Score
      - Macro-average Precision, Recall, F1
      - Confusion Matrix (10x10)
    """
    print("Evaluating Crop Health CNN on independent held-out test split...")
    cnn = CropHealthCNN()
    if not cnn.load_weights():
        raise FileNotFoundError(f"Model weights not found at {CNN_WEIGHTS_PATH}. Please run train_model.py first.")

    num_classes = len(CLASSES)
    X_test_list = []
    y_true_list = []

    for cls_idx, cls_name in enumerate(CLASSES):
        for s in range(test_samples_per_class):
            # Unseen seeds (5000+)
            seed = 5000 + cls_idx * 50 + s
            img_bgr = generate_image_for_class(cls_name, random_seed=seed)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            img_norm = img_resized.astype(np.float32) / 255.0

            X_test_list.append(img_norm)
            y_true_list.append(cls_idx)

    X_test = np.array(X_test_list, dtype=np.float32)
    y_true = np.array(y_true_list, dtype=np.int32)
    total_test = len(y_true)

    # Inference
    probs, _ = cnn.forward(X_test)
    y_pred = np.argmax(probs, axis=1)

    # Compute Confusion Matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    # Per-Class Metrics
    class_metrics = {}
    precisions = []
    recalls = []
    f1_scores = []

    for i, class_name in enumerate(CLASSES):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        class_metrics[class_name] = {
            "precision": round(float(prec) * 100.0, 2),
            "recall": round(float(rec) * 100.0, 2),
            "f1_score": round(float(f1) * 100.0, 2),
            "support": int(np.sum(confusion_matrix[i, :]))
        }

    overall_accuracy = float(np.sum(y_true == y_pred) / total_test) * 100.0
    macro_precision = float(np.mean(precisions)) * 100.0
    macro_recall = float(np.mean(recalls)) * 100.0
    macro_f1 = float(np.mean(f1_scores)) * 100.0

    eval_results = {
        "overall_accuracy": round(overall_accuracy, 2),
        "macro_precision": round(macro_precision, 2),
        "macro_recall": round(macro_recall, 2),
        "macro_f1": round(macro_f1, 2),
        "total_test_samples": total_test,
        "classes": CLASSES,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_metrics": class_metrics
    }

    out_file = MODEL_DIR / "eval_metrics.json"
    with open(out_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    print("=" * 60)
    print("AI Model Evaluation Summary:")
    print(f"Overall Accuracy:  {overall_accuracy:.2f}%")
    print(f"Macro Precision:   {macro_precision:.2f}%")
    print(f"Macro Recall:      {macro_recall:.2f}%")
    print(f"Macro F1-Score:    {macro_f1:.2f}%")
    print(f"Metrics saved to:  {out_file}")
    print("=" * 60)

    return eval_results

if __name__ == "__main__":
    evaluate_model()
