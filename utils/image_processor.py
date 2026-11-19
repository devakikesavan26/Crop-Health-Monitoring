"""
OpenCV Image Processing and Severity Analysis Module
AI-Based Crop Health Monitoring and Early Disease Detection System
"""

import cv2
import numpy as np
from PIL import Image
import io
import os
from config import IMAGE_SIZE

class ImageProcessor:
    """
    Handles image validation, OpenCV preprocessing, and lesion severity segmentation.
    """

    @staticmethod
    def validate_image(image_bytes_or_file):
        """
        Validates whether the provided file is a readable, non-corrupted image
        with sufficient resolution and contains organic vegetation/crop foliage.
        Returns: (is_valid: bool, message: str, cv2_bgr_img: np.ndarray)
        """
        if image_bytes_or_file is None:
            return False, "No image file was uploaded or captured.", None

        try:
            # Read bytes
            if hasattr(image_bytes_or_file, "read"):
                # Seek to beginning if needed
                image_bytes_or_file.seek(0)
                file_bytes = np.asarray(bytearray(image_bytes_or_file.read()), dtype=np.uint8)
                image_bytes_or_file.seek(0)
            elif isinstance(image_bytes_or_file, (bytes, bytearray)):
                file_bytes = np.asarray(bytearray(image_bytes_or_file), dtype=np.uint8)
            elif isinstance(image_bytes_or_file, str) and os.path.exists(image_bytes_or_file):
                with open(image_bytes_or_file, "rb") as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            else:
                return False, "Unsupported image input type.", None

            # Decode using OpenCV
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return False, "Corrupted or unreadable image format. Please upload a valid JPG or PNG file.", None

            h, w = img_bgr.shape[:2]
            if h < 32 or w < 32:
                return False, f"Image resolution ({w}x{h}) is too low for disease detection. Minimum resolution is 32x32.", None

            # Color Content Verification (Vegetation / Crop foliage check)
            # Plants have significant presence of chlorophyll (green) or senescence/necrosis (yellow/brown)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Green foliage mask: H: 25-90, S: 25-255, V: 25-255
            mask_green = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([90, 255, 255]))
            
            # Yellow/Brownish leaf/lesion mask: H: 10-25, S: 30-255, V: 30-250
            mask_yellow_brown = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([25, 255, 250]))
            
            plant_pixels = cv2.countNonZero(mask_green | mask_yellow_brown)
            total_pixels = h * w
            plant_ratio = plant_pixels / total_pixels

            # If the plant pixel ratio is very low, warn or check
            if plant_ratio < 0.05:
                return True, "Notice: Image contains very low vegetative color. Ensure the subject is a crop leaf.", img_bgr

            return True, "Image successfully validated.", img_bgr

        except Exception as e:
            return False, f"Error validating image: {str(e)}", None

    @staticmethod
    def preprocess_for_cnn(img_bgr, target_size=IMAGE_SIZE):
        """
        Resizes and normalizes the image for the Convolutional Neural Network.
        Returns: normalized batch tensor with shape (1, H, W, 3) in range [0, 1]
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize with high-quality interpolation
        resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
        # Normalize to float32 [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension (1, H, W, C)
        tensor = np.expand_dims(normalized, axis=0)
        return tensor, resized

    @staticmethod
    def analyze_severity(img_bgr, is_healthy=False):
        """
        Performs OpenCV computer-vision based severity analysis by segmenting:
        1. Leaf canopy boundary (Total Leaf Area)
        2. Lesions, necrosis, chlorosis, and rust spots (Diseased Area)
        
        Returns:
            severity_dict = {
                "lesion_percentage": float,
                "severity_level": "Healthy" | "Mild" | "Moderate" | "Severe",
                "leaf_area_pixels": int,
                "lesion_area_pixels": int,
                "visual_overlay_bgr": np.ndarray,
                "mask_overlay_bgr": np.ndarray
            }
        """
        h, w = img_bgr.shape[:2]
        
        # Convert to HSV and LAB color spaces for robust segmentation
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Segment Total Leaf Area
        # A leaf typically contrasts against background (neutral, soil, or hand)
        # Combined mask for healthy green, yellowing, and necrotic leaf tissue
        lower_leaf = np.array([10, 20, 20])
        upper_leaf = np.array([95, 255, 255])
        leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

        # Apply morphological operations to clean leaf mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

        leaf_pixel_count = cv2.countNonZero(leaf_mask)
        if leaf_pixel_count == 0:
            leaf_pixel_count = max(1, int(h * w * 0.5))
            leaf_mask = np.ones((h, w), dtype=np.uint8) * 255

        if is_healthy:
            # For confirmed healthy leaves, severity is 0%
            lesion_percentage = 0.0
            severity_level = "Healthy (0%)"
            overlay_bgr = img_bgr.copy()
            # Draw a subtle healthy green contour around the leaf
            contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_bgr, contours, -1, (0, 200, 0), 2)
            
            return {
                "lesion_percentage": 0.0,
                "severity_level": severity_level,
                "leaf_area_pixels": leaf_pixel_count,
                "lesion_area_pixels": 0,
                "visual_overlay_bgr": overlay_bgr,
                "mask_overlay_bgr": cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)
            }

        # 2. Segment Diseased / Lesion Area within the leaf boundary
        # Lesions manifest as:
        # - Necrotic brown/black dead tissue: Low V or low H (0-20), medium to high S
        # - Chlorotic yellow halos: H (15-28), high S
        # - Rust pustules: H (8-22), S (80-255), V (50-200)
        
        # Necrotic mask
        lower_necrotic = np.array([5, 35, 20])
        upper_necrotic = np.array([22, 255, 180])
        mask_necrotic = cv2.inRange(hsv, lower_necrotic, upper_necrotic)

        # Dark spot / black mold mask
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 65])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        # Yellow halo chlorosis mask
        lower_halo = np.array([20, 70, 70])
        upper_halo = np.array([32, 255, 255])
        mask_halo = cv2.inRange(hsv, lower_halo, upper_halo)

        # Combined lesion mask, strictly bounded within leaf mask
        raw_lesion_mask = (mask_necrotic | mask_dark | mask_halo) & leaf_mask

        # Clean lesion noise with opening
        lesion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_lesion_mask = cv2.morphologyEx(raw_lesion_mask, cv2.MORPH_OPEN, lesion_kernel)

        lesion_pixel_count = cv2.countNonZero(clean_lesion_mask)
        
        # Calculate Infection Percentage
        lesion_percentage = min(100.0, (lesion_pixel_count / float(leaf_pixel_count)) * 100.0)

        # Categorize Severity Level
        if lesion_percentage < 1.0:
            severity_level = "Healthy / Trace (<1%)"
        elif lesion_percentage <= 12.0:
            severity_level = "Mild (Early Stage)"
        elif lesion_percentage <= 30.0:
            severity_level = "Moderate (Active Spread)"
        else:
            severity_level = "Severe (Critical Threat)"

        # 3. Create High-Contrast Visual Overlay for Farmers
        overlay_bgr = img_bgr.copy()
        
        # Highlight lesions in bright red/crimson overlay
        colored_mask = np.zeros_like(img_bgr)
        colored_mask[clean_lesion_mask > 0] = [0, 0, 235]  # Red in BGR

        # Blend with original image
        cv2.addWeighted(colored_mask, 0.45, overlay_bgr, 0.55, 0, overlay_bgr)

        # Draw red contours around discrete lesion clusters
        lesion_contours, _ = cv2.findContours(clean_lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, lesion_contours, -1, (0, 0, 255), 2)

        # Draw leaf outline in green
        leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, leaf_contours, -1, (0, 255, 0), 2)

        # Create diagnostic comparison mask
        diagnostic_mask = np.zeros_like(img_bgr)
        diagnostic_mask[leaf_mask > 0] = [60, 140, 60]       # Green for healthy leaf area
        diagnostic_mask[clean_lesion_mask > 0] = [0, 69, 255] # Orange-Red for lesion area

        return {
            "lesion_percentage": round(lesion_percentage, 1),
            "severity_level": severity_level,
            "leaf_area_pixels": leaf_pixel_count,
            "lesion_area_pixels": lesion_pixel_count,
            "visual_overlay_bgr": overlay_bgr,
            "mask_overlay_bgr": diagnostic_mask
        }
