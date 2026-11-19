"""
Agricultural Leaf Dataset and Sample Generator
AI-Based Crop Health Monitoring and Early Disease Detection System
Generates botanical leaf patterns for training and interactive testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import os
from config import CLASSES, SAMPLE_IMAGES_DIR

def create_leaf_canvas(width=256, height=256, bg_color=(235, 235, 235)):
    """Creates a neutral background canvas for realistic leaf rendering."""
    canvas = np.ones((height, width, 3), dtype=np.uint8)
    for c in range(3):
        canvas[:, :, c] = bg_color[c]
    # Add slight background texture noise
    noise = np.random.randint(-5, 6, (height, width, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas

def draw_leaf_base(canvas, leaf_type="ovate", base_color=(45, 140, 50)):
    """Draws realistic leaf morphology with veins."""
    h, w = canvas.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2

    if leaf_type == "ovate": # Pepper / Tomato leaflet
        pts = np.array([
            [center_x, 30],
            [center_x + 65, 80],
            [center_x + 85, 140],
            [center_x + 55, 200],
            [center_x, 225],
            [center_x - 55, 200],
            [center_x - 85, 140],
            [center_x - 65, 80]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    elif leaf_type == "compound": # Tomato / Potato
        # Main central leaflet
        pts_main = np.array([
            [center_x, 35],
            [center_x + 55, 85],
            [center_x + 70, 135],
            [center_x + 40, 185],
            [center_x, 215],
            [center_x - 40, 185],
            [center_x - 70, 135],
            [center_x - 55, 85]
        ], np.int32)
        cv2.fillPoly(mask, [pts_main], 255)
        # Side sub-leaflets
        cv2.ellipse(mask, (center_x - 50, 160), (35, 18), 35, 0, 360, 255, -1)
        cv2.ellipse(mask, (center_x + 50, 160), (35, 18), -35, 0, 360, 255, -1)
    elif leaf_type == "linear": # Maize / Corn
        pts_corn = np.array([
            [center_x, 20],
            [center_x + 35, 80],
            [center_x + 45, 150],
            [center_x + 35, 210],
            [center_x + 20, 240],
            [center_x - 20, 240],
            [center_x - 35, 210],
            [center_x - 45, 150],
            [center_x - 35, 80]
        ], np.int32)
        cv2.fillPoly(mask, [pts_corn], 255)

    # Smooth the leaf contour
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Fill base color with natural shading
    b, g, r = base_color
    leaf_rgb = np.zeros_like(canvas)
    for y in range(h):
        # subtle vertical gradient (leaf tip slightly brighter green)
        shade_factor = 0.85 + 0.3 * (y / float(h))
        leaf_rgb[y, :, 0] = np.clip(b * shade_factor, 0, 255)
        leaf_rgb[y, :, 1] = np.clip(g * shade_factor, 0, 255)
        leaf_rgb[y, :, 2] = np.clip(r * shade_factor, 0, 255)

    # Draw Central Midrib Vein
    cv2.line(leaf_rgb, (center_x, 30), (center_x, 240), (b - 15, g + 25, r - 15), 3)

    # Draw Lateral Secondary Veins
    if leaf_type != "linear":
        for vy in range(70, 210, 25):
            cv2.line(leaf_rgb, (center_x, vy), (center_x - 55, vy - 20), (b - 10, g + 15, r - 10), 1)
            cv2.line(leaf_rgb, (center_x, vy), (center_x + 55, vy - 20), (b - 10, g + 15, r - 10), 1)
    else:
        # Parallel veins for monocot corn
        for offset in [-25, -15, -8, 8, 15, 25]:
            cv2.line(leaf_rgb, (center_x + offset, 40), (center_x + offset, 235), (b - 8, g + 12, r - 8), 1)

    # Blend leaf onto canvas
    canvas[mask > 0] = leaf_rgb[mask > 0]
    return canvas, mask

def apply_early_blight_symptoms(canvas, mask, num_spots=5):
    """Draws target-board concentric ring spots with yellow chlorotic halo."""
    h, w = canvas.shape[:2]
    leaf_indices = np.where(mask > 0)
    if len(leaf_indices[0]) == 0:
        return canvas

    for _ in range(num_spots):
        idx = np.random.randint(0, len(leaf_indices[0]))
        cy = leaf_indices[0][idx]
        cx = leaf_indices[1][idx]
        radius = np.random.randint(12, 22)

        # 1. Yellow chlorotic halo around lesion
        cv2.circle(canvas, (cx, cy), radius + 7, (40, 200, 220), -1)  # Yellow-orange in BGR
        # 2. Dark brown necrotic outer rim
        cv2.circle(canvas, (cx, cy), radius, (20, 45, 90), -1)
        # 3. Concentric target rings
        cv2.circle(canvas, (cx, cy), int(radius * 0.7), (25, 60, 115), 2)
        cv2.circle(canvas, (cx, cy), int(radius * 0.45), (15, 35, 75), 2)
        # 4. Central dark necrotic core
        cv2.circle(canvas, (cx, cy), int(radius * 0.2), (10, 20, 50), -1)

    # Keep only inside leaf mask
    return canvas

def apply_late_blight_symptoms(canvas, mask, num_lesions=3):
    """Draws water-soaked large dark olive-brown spreading necrotic lesions."""
    h, w = canvas.shape[:2]
    leaf_indices = np.where(mask > 0)
    if len(leaf_indices[0]) == 0:
        return canvas

    for _ in range(num_lesions):
        idx = np.random.randint(0, len(leaf_indices[0]))
        cy = leaf_indices[0][idx]
        cx = leaf_indices[1][idx]
        axes = (np.random.randint(22, 38), np.random.randint(15, 28))
        angle = np.random.randint(0, 180)

        # Pale grayish-yellow water-soaked perimeter
        cv2.ellipse(canvas, (cx, cy), (axes[0] + 6, axes[1] + 6), angle, 0, 360, (50, 160, 175), -1)
        # Dark blackish-brown decaying core
        cv2.ellipse(canvas, (cx, cy), axes, angle, 0, 360, (18, 30, 45), -1)
        # Subtle fungal fuzz rim
        cv2.ellipse(canvas, (cx, cy), (axes[0] + 2, axes[1] + 2), angle, 0, 360, (190, 200, 205), 1)

    return canvas

def apply_bacterial_spot_symptoms(canvas, mask, num_spots=18):
    """Draws small angular water-soaked dark spots with yellow halos."""
    h, w = canvas.shape[:2]
    leaf_indices = np.where(mask > 0)
    if len(leaf_indices[0]) == 0:
        return canvas

    for _ in range(num_spots):
        idx = np.random.randint(0, len(leaf_indices[0]))
        cy = leaf_indices[0][idx]
        cx = leaf_indices[1][idx]
        rad = np.random.randint(3, 7)
        # Yellow halo
        cv2.circle(canvas, (cx, cy), rad + 3, (30, 180, 215), -1)
        # Dark necrotic center
        cv2.circle(canvas, (cx, cy), rad, (15, 25, 40), -1)

    return canvas

def apply_common_rust_symptoms(canvas, mask, num_pustules=24):
    """Draws elongated cinnamon-brown to golden-brown powdery pustules."""
    h, w = canvas.shape[:2]
    leaf_indices = np.where(mask > 0)
    if len(leaf_indices[0]) == 0:
        return canvas

    for _ in range(num_pustules):
        idx = np.random.randint(0, len(leaf_indices[0]))
        cy = leaf_indices[0][idx]
        cx = leaf_indices[1][idx]
        # Elongated oval along the leaf axis
        axes = (np.random.randint(3, 6), np.random.randint(8, 16))
        angle = np.random.randint(-10, 10)
        # Golden yellow border
        cv2.ellipse(canvas, (cx, cy), (axes[0] + 2, axes[1] + 2), angle, 0, 360, (20, 140, 210), -1)
        # Cinnamon-brown powdery rust pustule
        cv2.ellipse(canvas, (cx, cy), axes, angle, 0, 360, (15, 65, 160), -1)

    return canvas

def generate_image_for_class(class_name, random_seed=None):
    """Generates an image for any of the 10 supported classes."""
    if random_seed is not None:
        np.random.seed(random_seed)

    canvas = create_leaf_canvas(256, 256)
    
    if "Tomato" in class_name:
        leaf_type = "compound"
        base_color = (35, 135, 45) # Tomato green
    elif "Potato" in class_name:
        leaf_type = "compound"
        base_color = (30, 125, 40) # Deep potato green
    elif "Pepper" in class_name:
        leaf_type = "ovate"
        base_color = (40, 145, 55) # Glossy capsicum green
    elif "Corn" in class_name:
        leaf_type = "linear"
        base_color = (45, 150, 60) # Maize green
    else:
        leaf_type = "ovate"
        base_color = (40, 130, 50)

    canvas, mask = draw_leaf_base(canvas, leaf_type=leaf_type, base_color=base_color)

    # Apply disease specific symptoms
    if "Early_blight" in class_name:
        canvas = apply_early_blight_symptoms(canvas, mask, num_spots=np.random.randint(4, 8))
    elif "Late_blight" in class_name:
        canvas = apply_late_blight_symptoms(canvas, mask, num_lesions=np.random.randint(3, 6))
    elif "Bacterial_spot" in class_name:
        canvas = apply_bacterial_spot_symptoms(canvas, mask, num_spots=np.random.randint(15, 26))
    elif "Common_rust" in class_name:
        canvas = apply_common_rust_symptoms(canvas, mask, num_pustules=np.random.randint(20, 32))
    elif "healthy" in class_name:
        # Healthy remains clean and green!
        pass

    return canvas

def generate_sample_images():
    """Generates and writes test sample images to sample_images/ directory."""
    os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
    
    samples = {
        "tomato_healthy.jpg": ("Tomato___healthy", 101),
        "tomato_early_blight.jpg": ("Tomato___Early_blight", 102),
        "tomato_late_blight.jpg": ("Tomato___Late_blight", 103),
        "potato_healthy.jpg": ("Potato___healthy", 104),
        "potato_early_blight.jpg": ("Potato___Early_blight", 105),
        "potato_late_blight.jpg": ("Potato___Late_blight", 106),
        "pepper_bell_healthy.jpg": ("Pepper__bell___healthy", 107),
        "pepper_bacterial_spot.jpg": ("Pepper__bell___Bacterial_spot", 108),
        "corn_healthy.jpg": ("Corn_(maize)___healthy", 109),
        "corn_common_rust.jpg": ("Corn_(maize)___Common_rust_", 110),
    }

    generated_paths = {}
    for filename, (cls_name, seed) in samples.items():
        img = generate_image_for_class(cls_name, random_seed=seed)
        out_path = SAMPLE_IMAGES_DIR / filename
        cv2.imwrite(str(out_path), img)
        generated_paths[cls_name] = str(out_path)

    # Generate an unrelated non-leaf test image (for rejection test)
    non_leaf = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(non_leaf, (30, 30), (226, 226), (80, 80, 80), -1)
    cv2.putText(non_leaf, "UNRELATED OBJECT", (35, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    non_leaf_path = SAMPLE_IMAGES_DIR / "non_leaf_sample.jpg"
    cv2.imwrite(str(non_leaf_path), non_leaf)
    
    print(f"Generated {len(samples) + 1} sample images in {SAMPLE_IMAGES_DIR}")
    return generated_paths

if __name__ == "__main__":
    generate_sample_images()
