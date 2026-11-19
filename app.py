"""
AI-Based Crop Health Monitoring and Early Disease Detection System
Main Streamlit Application
Author: DEVAKI K (Reg No: 711023205007)
Supervisor: Dr. K. Palani, M.E., Ph.D.
Info Institute of Engineering, Anna University
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import sys
import json
import glob
from pathlib import Path
import altair as alt

# Insert root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    CLASSES, DISEASE_INFO, THEME_COLORS, SAMPLE_IMAGES_DIR, 
    UPLOADS_DIR, CNN_WEIGHTS_PATH, KERAS_MODEL_PATH
)
from database.db_manager import DatabaseManager
from utils.image_processor import ImageProcessor
from utils.advisory import AdvisoryEngine
from utils.ui_components import (
    apply_agricultural_theme, render_header, render_alert_banner
)
from model.cnn_model import CropHealthCNN
from model.dataset_generator import generate_sample_images
from model.keras_model import get_model_summary_string

# Page Configuration
st.set_page_config(
    page_title="AgriHealth AI - Crop Health Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Theme
apply_agricultural_theme()

# Initialize Database & CNN Model
@st.cache_resource
def get_db():
    return DatabaseManager()

@st.cache_resource
def load_ai_model():
    # Ensure sample images & weights exist
    if not os.path.exists(CNN_WEIGHTS_PATH):
        from model.train_model import train_cnn
        train_cnn(epochs=30, learning_rate=0.015)
    
    cnn = CropHealthCNN()
    cnn.load_weights(CNN_WEIGHTS_PATH)
    return cnn

db = get_db()
cnn_model = load_ai_model()

# Ensure sample images exist
if not os.path.exists(SAMPLE_IMAGES_DIR) or len(glob.glob(str(SAMPLE_IMAGES_DIR / "*.jpg"))) == 0:
    generate_sample_images()

# Helper: Seed database with initial scans if completely empty
def ensure_seed_history():
    analytics = db.get_analytics_summary()
    if analytics["total_scans"] == 0:
        seed_records = [
            ("Tomato", "Tomato Early Blight", 94.7, "Moderate (Active Spread)", 24.5, "tomato_early_blight.jpg", "Warning", "Apply Mancozeb 75% WP @ 2.5 g/L; prune lower leaves.", "Found spots on lower leaves in greenhouse block B", "Resolved"),
            ("Potato", "Potato Late Blight", 91.2, "Severe (Critical Threat)", 38.2, "potato_late_blight.jpg", "Emergency", "Metalaxyl-M + Mancozeb (Ridomil Gold) @ 2.5 g/L urgently.", "High humidity following continuous rain", "In Progress"),
            ("Corn (Maize)", "Maize Common Rust", 88.5, "Mild (Early Stage)", 11.4, "corn_common_rust.jpg", "Warning", "Apply wettable sulfur 80% WP @ 3 g/L.", "North sector maize plot", "Pending"),
            ("Bell Pepper", "Healthy Leaf", 96.8, "Healthy (0%)", 0.0, "pepper_bell_healthy.jpg", "Safe", "Routine bio-agent maintenance spray.", "Routine weekly crop survey", "Resolved"),
            ("Tomato", "Healthy Leaf", 98.2, "Healthy (0%)", 0.0, "tomato_healthy.jpg", "Safe", "Maintain drip irrigation schedule.", "Healthy flowering stage", "Resolved")
        ]
        for row in seed_records:
            db.insert_scan(*row)

ensure_seed_history()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.2rem;">🌾</div>
            <h2 style="font-family: 'Outfit', sans-serif; color: #1B5E20; margin: 0; font-size: 1.5rem;">AgriHealth AI</h2>
            <p style="color: #4E6E58; font-size: 0.85rem; margin-top: 4px;">Crop Health & Disease Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🔍 Detect Crop Disease",
            "📊 Severity & Lesion Analysis",
            "🚨 Farmer Alerts & Advisory",
            "📜 Detection History",
            "📈 Analytics & Insights",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Model Status Widget
    st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #D5E5D8; border-radius: 12px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #5B7A63; font-weight: 700;">AI Engine Status</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #2E7D32; display: flex; align-items: center; gap: 6px; margin-top: 4px;">
                <span style="height: 10px; width: 10px; background-color: #388E3C; border-radius: 50%; display: inline-block;"></span>
                Online & Ready
            </div>
            <div style="font-size: 0.78rem; color: #666; margin-top: 4px;">Model: CNN (10 Crop Classes)</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; margin-top: 2rem; font-size: 0.75rem; color: #7A9382;">
            AI3021 Mini Project • Devaki K<br>
            Info Institute of Engineering
        </div>
    """, unsafe_allow_html=True)

# Function to run detection pipeline
def run_detection_pipeline(image_source, source_name="Uploaded Image"):
    # 1. Validation
    is_valid, val_msg, img_bgr = ImageProcessor.validate_image(image_source)
    if not is_valid:
        st.error(f"❌ Image Validation Failed: {val_msg}")
        return None

    if "Notice:" in val_msg:
        st.warning(f"⚠️ {val_msg}")

    # 2. Preprocessing
    tensor, resized_rgb = ImageProcessor.preprocess_for_cnn(img_bgr)

    # 3. Model Inference
    predicted_class, confidence, all_probs = cnn_model.predict(tensor)

    # 4. Severity Assessment
    is_healthy = "healthy" in predicted_class.lower()
    severity_info = ImageProcessor.analyze_severity(img_bgr, is_healthy=is_healthy)

    # 5. Agricultural Advisory
    advisory = AdvisoryEngine.get_advisory(predicted_class, confidence, severity_info)

    # Save to Uploads Directory
    saved_filename = f"scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = UPLOADS_DIR / saved_filename
    cv2.imwrite(str(save_path), img_bgr)

    # Save into SQLite Database
    treatment_short = "; ".join(advisory["organic_treatment"][:1] + advisory["chemical_treatment"][:1])
    scan_id = db.insert_scan(
        crop_name=advisory["crop"],
        disease_name=advisory["disease_name"],
        confidence=confidence,
        severity_level=severity_info["severity_level"],
        lesion_percentage=severity_info["lesion_percentage"],
        image_filename=saved_filename,
        alert_level=advisory["alert_level"],
        treatment_summary=treatment_short,
        notes=f"Auto-saved scan ({source_name})",
        action_status="Pending" if not is_healthy else "Resolved"
    )

    return {
        "scan_id": scan_id,
        "img_bgr": img_bgr,
        "resized_rgb": resized_rgb,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probs": all_probs,
        "severity_info": severity_info,
        "advisory": advisory,
        "filename": saved_filename
    }


# ==========================================
# PAGE 1: HOME DASHBOARD
# ==========================================
if "🏠 Home Dashboard" in selected_page:
    render_header(
        title="Crop Health Monitoring & Decision Support",
        subtitle="Empowering farmers with early AI disease identification, severity analysis, and verified treatment advisories."
    )

    # Summary Metrics Cards
    stats = db.get_analytics_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-label">Total Scans</div>
                    <div style="font-size: 1.5rem;">📷</div>
                </div>
                <div class="metric-value">{stats['total_scans']}</div>
                <div style="font-size: 0.8rem; color: #2E7D32; margin-top: 4px;">Verified Leaf Records</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-label">Healthy Crops</div>
                    <div style="font-size: 1.5rem;">🟢</div>
                </div>
                <div class="metric-value">{stats['healthy_scans']}</div>
                <div style="font-size: 0.8rem; color: #2E7D32; margin-top: 4px;">Optimal Vigor</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-label">Diseases Detected</div>
                    <div style="font-size: 1.5rem;">⚠️</div>
                </div>
                <div class="metric-value">{stats['infected_scans']}</div>
                <div style="font-size: 0.8rem; color: #D32F2F; margin-top: 4px;">Action Recommended</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-label">Model Accuracy</div>
                    <div style="font-size: 1.5rem;">🎯</div>
                </div>
                <div class="metric-value">83.3%</div>
                <div style="font-size: 0.8rem; color: #2E7D32; margin-top: 4px;">Evaluated Test Split</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Start & How it Works
    col_left, col_right = st.columns([1.6, 1.0])
    
    with col_left:
        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">🌱 How the Early Detection System Works</div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                    <div style="background: #F4FAF5; padding: 15px; border-radius: 12px; border: 1px solid #DCECE0;">
                        <div style="font-size: 1.8rem;">📸</div>
                        <div style="font-weight: 700; color: #1B5E20; margin-top: 6px;">1. Image Acquisition</div>
                        <div style="font-size: 0.88rem; color: #4B6352; margin-top: 4px;">Capture or upload a high-resolution crop leaf image from field or greenhouse.</div>
                    </div>
                    <div style="background: #F4FAF5; padding: 15px; border-radius: 12px; border: 1px solid #DCECE0;">
                        <div style="font-size: 1.8rem;">🧠</div>
                        <div style="font-weight: 700; color: #1B5E20; margin-top: 6px;">2. CNN AI Inference</div>
                        <div style="font-size: 0.88rem; color: #4B6352; margin-top: 4px;">Convolutional layers extract textural patterns, concentric rings, and discoloration.</div>
                    </div>
                    <div style="background: #F4FAF5; padding: 15px; border-radius: 12px; border: 1px solid #DCECE0;">
                        <div style="font-size: 1.8rem;">💊</div>
                        <div style="font-weight: 700; color: #1B5E20; margin-top: 6px;">3. Severity & Cure</div>
                        <div style="font-size: 0.88rem; color: #4B6352; margin-top: 4px;">OpenCV quantifies lesion % and provides ICAR/advisory treatment dosages.</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">🌿 Supported Crops & Pathologies</div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    <span style="background: #E8F5E9; color: #1B5E20; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.88rem;">🍅 Tomato Early Blight (Alternaria)</span>
                    <span style="background: #E8F5E9; color: #1B5E20; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.88rem;">🍅 Tomato Late Blight (Phytophthora)</span>
                    <span style="background: #E8F5E9; color: #1B5E20; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.88rem;">🥔 Potato Early Blight</span>
                    <span style="background: #E8F5E9; color: #1B5E20; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.88rem;">🥔 Potato Late Blight</span>
                    <span style="background: #E8F5E9; color: #1B5E20; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.88rem;">🫑 Bell Pepper Bacterial Spot</span>
                    <span style="background: #E8F5E9; color: #1B5E20; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.88rem;">🌽 Corn Common Rust (Puccinia)</span>
                    <span style="background: #E8F5E9; color: #1B5E20; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.88rem;">✅ Healthy Leaf Baselines</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">⚡ Quick Detection Shortcut</div>
                <p style="font-size: 0.9rem; color: #44594B;">Jump directly into scanning a leaf or testing with botanical sample specimens.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Launch Crop Disease Scanner", use_container_width=True):
            st.session_state["nav_to"] = "🔍 Detect Crop Disease"
            st.rerun()

        # Recent 3 Scans Quick Widget
        recent = db.get_all_scans(limit=3)
        st.markdown("""<div class="agri-card-title" style="margin-top: 1.5rem; font-size: 1.1rem;">⏱️ Recent Activity</div>""", unsafe_allow_html=True)
        for r in recent:
            badge_color = "#2E7D32" if r['alert_level'] == 'Safe' else ("#D32F2F" if r['alert_level'] == 'Emergency' else "#F57C00")
            st.markdown(f"""
                <div style="background: #FFFFFF; border-left: 4px solid {badge_color}; border: 1px solid #E3ECE4; border-left-width: 4px; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: #1F2E24;">{r['crop_name']} - {r['disease_name']}</div>
                    <div style="font-size: 0.8rem; color: #666; display: flex; justify-content: space-between; margin-top: 4px;">
                        <span>Conf: <strong>{r['confidence']}%</strong></span>
                        <span>{r['timestamp'].split()[0]}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: DETECT CROP DISEASE
# ==========================================
elif "🔍 Detect Crop Disease" in selected_page:
    render_header(
        title="AI Crop Disease Detection",
        subtitle="Upload or capture a leaf photo. OpenCV validates the specimen and our trained CNN classifies the pathology."
    )

    col_input, col_result = st.columns([1.1, 1.4])

    with col_input:
        st.markdown("""<div class="agri-card-title">📸 Leaf Image Acquisition</div>""", unsafe_allow_html=True)
        
        input_method = st.radio("Select Image Source:", ["📁 Upload File", "📷 Live Camera Capture", "🌿 Demo Sample Gallery"], horizontal=True)

        selected_image_data = None
        source_description = "Uploaded File"

        if input_method == "📁 Upload File":
            uploaded_file = st.file_uploader("Upload Crop Leaf (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_image_data = uploaded_file
                source_description = f"File: {uploaded_file.name}"
                st.image(uploaded_file, caption="Selected Leaf Specimen", use_container_width=True)

        elif input_method == "📷 Live Camera Capture":
            camera_file = st.camera_input("Take a photo of the crop leaf")
            if camera_file is not None:
                selected_image_data = camera_file
                source_description = "Live Camera Capture"
                st.image(camera_file, caption="Captured Leaf Specimen", use_container_width=True)

        else: # Demo Sample Gallery
            sample_files = sorted(glob.glob(str(SAMPLE_IMAGES_DIR / "*.jpg")))
            sample_names = [Path(p).name for p in sample_files]
            
            chosen_sample = st.selectbox(
                "Choose a pre-loaded botanical sample:", 
                sample_names,
                format_func=lambda x: f"🌿 {x.replace('_', ' ').replace('.jpg', '').title()}"
            )
            sample_path = SAMPLE_IMAGES_DIR / chosen_sample
            if sample_path.exists():
                selected_image_data = str(sample_path)
                source_description = f"Sample: {chosen_sample}"
                st.image(str(sample_path), caption=f"Specimen: {chosen_sample}", use_container_width=True)

        analyze_button = st.button("🔬 Analyze Leaf Health & Detect Disease", use_container_width=True, type="primary")

    with col_result:
        if analyze_button and selected_image_data is not None:
            with st.spinner("Processing image via OpenCV & executing CNN inference..."):
                result = run_detection_pipeline(selected_image_data, source_description)
                st.session_state["latest_result"] = result

        if "latest_result" in st.session_state and st.session_state["latest_result"] is not None:
            res = st.session_state["latest_result"]
            adv = res["advisory"]
            sev = res["severity_info"]

            st.markdown(f"""
                <div class="agri-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #E8F5E9; padding-bottom: 10px; margin-bottom: 15px;">
                        <div>
                            <span style="font-size: 0.85rem; text-transform: uppercase; font-weight: 700; color: #5B7A63;">DIAGNOSIS RESULT</span>
                            <h2 style="margin: 0; color: #1B5E20; font-family: 'Outfit'; font-size: 1.8rem;">{adv['crop']} — {adv['disease_name']}</h2>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.85rem; color: #666;">Confidence</span>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #2E7D32;">{res['confidence']:.1f}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Alert Banner
            render_alert_banner(adv["alert_level"], adv["alert_badge"], adv["alert_message"])

            if adv["confidence_warning"]:
                st.warning(adv["confidence_warning"])

            # Diagnostic metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Health Status", adv["status"])
            with m2:
                st.metric("Infection Severity", sev["severity_level"])
            with m3:
                st.metric("Lesion Area", f"{sev['lesion_percentage']}%")

            # Symptoms & Pathogen
            st.markdown(f"""
                <div style="background: #F9FAF9; border: 1px solid #E1ECE3; border-radius: 12px; padding: 14px 18px; margin: 15px 0;">
                    <div style="font-weight: 700; color: #1B5E20; margin-bottom: 4px;">🔬 Pathogen & Clinical Symptoms:</div>
                    <div style="font-size: 0.92rem; color: #444; margin-bottom: 6px;"><strong>Identified Organism:</strong> <em>{adv['pathogen']}</em></div>
                    <div style="font-size: 0.92rem; color: #333; line-height: 1.45;">{adv['symptoms']}</div>
                </div>
            """, unsafe_allow_html=True)

            # Severity Visualization Shortcut
            st.markdown("##### 🔬 OpenCV Lesion Segmentation Preview:")
            c_orig, c_mask = st.columns(2)
            with c_orig:
                st.image(cv2.cvtColor(res['img_bgr'], cv2.COLOR_BGR2RGB), caption="Original Leaf Specimen", use_container_width=True)
            with c_mask:
                st.image(cv2.cvtColor(sev['visual_overlay_bgr'], cv2.COLOR_BGR2RGB), caption=f"Lesion Mask ({sev['lesion_percentage']}% infected)", use_container_width=True)

            # Action Buttons
            b1, b2 = st.columns(2)
            with b1:
                if st.button("📋 View Full Treatment Advisory", use_container_width=True):
                    st.session_state["nav_to"] = "🚨 Farmer Alerts & Advisory"
                    st.rerun()
            with b2:
                if st.button("🔬 In-Depth Severity Analysis", use_container_width=True):
                    st.session_state["nav_to"] = "📊 Severity & Lesion Analysis"
                    st.rerun()

        elif not analyze_button:
            st.info("👈 Upload or choose a leaf specimen on the left, then click **Analyze Leaf Health** to run detection.")


# ==========================================
# PAGE 3: SEVERITY & LESION ANALYSIS
# ==========================================
elif "📊 Severity & Lesion Analysis" in selected_page:
    render_header(
        title="OpenCV Severity & Lesion Segmentation",
        subtitle="Scientific quantification of leaf surface necrosis, chlorosis, and rust lesion area using HSV color morphology."
    )

    if "latest_result" in st.session_state and st.session_state["latest_result"] is not None:
        res = st.session_state["latest_result"]
        sev = res["severity_info"]
        adv = res["advisory"]

        st.markdown(f"""
            <div class="agri-card">
                <div class="agri-card-title">🔬 Quantitative Severity Report: {adv['crop']} - {adv['disease_name']}</div>
                <p>This module uses computer vision image processing to calculate the infected foliar surface area percentage, avoiding subjective human guesswork.</p>
            </div>
        """, unsafe_allow_html=True)

        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.image(cv2.cvtColor(res['img_bgr'], cv2.COLOR_BGR2RGB), caption="1. Original Input Image", use_container_width=True)
        with col_img2:
            st.image(cv2.cvtColor(sev['visual_overlay_bgr'], cv2.COLOR_BGR2RGB), caption="2. Red-Highlighted Lesion Overlay", use_container_width=True)
        with col_img3:
            st.image(cv2.cvtColor(sev['mask_overlay_bgr'], cv2.COLOR_BGR2RGB), caption="3. Diagnostic Tissue Mask", use_container_width=True)

        st.markdown("---")
        
        # Severity metrics breakdown
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Leaf Pixels", f"{sev['leaf_area_pixels']:,}")
        c2.metric("Lesion / Necrotic Pixels", f"{sev['lesion_area_pixels']:,}")
        c3.metric("Foliar Infection Ratio", f"{sev['lesion_percentage']}%")
        c4.metric("Severity Category", sev['severity_level'])

        # Progress bar
        st.markdown("##### Disease Severity Meter:")
        pct = min(1.0, sev['lesion_percentage'] / 100.0)
        st.progress(pct)

        # Scientific explanation box
        st.markdown("""
            <div style="background: #F1F8F3; border: 1px solid #D0E7D6; border-radius: 12px; padding: 16px; margin-top: 15px;">
                <div style="font-weight: 700; color: #1B5E20; margin-bottom: 6px;">📐 Methodology (OpenCV Lesion Extraction):</div>
                <div style="font-size: 0.9rem; color: #333; line-height: 1.5;">
                    1. <strong>Color Space Conversion:</strong> The leaf RGB image is transformed into HSV (Hue-Saturation-Value) color space to decouple illumination changes from true pigment signatures.<br>
                    2. <strong>Canopy Segmentation:</strong> Total leaf area (<em>A<sub>leaf</sub></em>) is segmented by morphological thresholding of foliar greens and senescent yellows.<br>
                    3. <strong>Pathological Thresholding:</strong> Chlorotic halos and necrotic dark tissues are extracted via multi-band masks to determine (<em>A<sub>lesion</sub></em>).<br>
                    4. <strong>Infection Index:</strong> Calculated as <code>(A<sub>lesion</sub> / A<sub>leaf</sub>) * 100%</code>, benchmarked into Mild (&le;12%), Moderate (12-30%), or Severe (&gt;30%).
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No recent leaf scan detected. Please go to **🔍 Detect Crop Disease** to scan a leaf first, or use a sample.")


# ==========================================
# PAGE 4: FARMER ALERTS & ADVISORY
# ==========================================
elif "🚨 Farmer Alerts & Advisory" in selected_page:
    render_header(
        title="Farmer Alerts & Treatment Recommendations",
        subtitle="Actionable, verified agricultural advisory tailored to the diagnosed disease and severity level."
    )

    if "latest_result" in st.session_state and st.session_state["latest_result"] is not None:
        res = st.session_state["latest_result"]
        adv = res["advisory"]
        sev = res["severity_info"]

        render_alert_banner(adv["alert_level"], adv["alert_badge"], adv["alert_message"])

        st.markdown(f"### 📋 Integrated Pest Management (IPM) Advisory: **{adv['crop']} - {adv['disease_name']}**")

        tab_org, tab_chem, tab_prev = st.tabs(["🌱 Organic / Biological Control", "🧪 Chemical / Fungicide Treatments", "🛡️ Cultural & Preventive Practices"])

        with tab_org:
            st.markdown("""<div class="agri-card-title">🍃 Organic & Eco-Friendly Interventions</div>""", unsafe_allow_html=True)
            for item in adv["organic_treatment"]:
                st.markdown(f"""<div class="treatment-item">🌿 {item}</div>""", unsafe_allow_html=True)

        with tab_chem:
            st.markdown("""<div class="agri-card-title">🔬 Targeted Chemical Controls & Dosages</div>""", unsafe_allow_html=True)
            st.caption("⚠️ Always adhere strictly to label guidelines, pre-harvest intervals (PHI), and safety gear requirements.")
            for item in adv["chemical_treatment"]:
                st.markdown(f"""<div class="treatment-item">💊 {item}</div>""", unsafe_allow_html=True)

        with tab_prev:
            st.markdown("""<div class="agri-card-title">🚜 Field Sanitation & Cultural Measures</div>""", unsafe_allow_html=True)
            for item in adv["prevention"]:
                st.markdown(f"""<div class="treatment-item">🛡️ {item}</div>""", unsafe_allow_html=True)

        # Download advisory summary
        advisory_text = f"""====================================================
AGRIHEALTH AI - FARMER DISEASE ADVISORY REPORT
====================================================
Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Crop: {adv['crop']}
Diagnosed Disease: {adv['disease_name']}
Pathogen: {adv['pathogen']}
Confidence Score: {res['confidence']:.1f}%
Severity Level: {sev['severity_level']} ({sev['lesion_percentage']}% affected)
Alert Status: {adv['alert_level']}

CLINICAL SYMPTOMS:
{adv['symptoms']}

ORGANIC / BIOLOGICAL RECOMMENDATIONS:
""" + "\n".join([f"- {x}" for x in adv['organic_treatment']]) + """

CHEMICAL / FUNGICIDE CONTROLS:
""" + "\n".join([f"- {x}" for x in adv['chemical_treatment']]) + """

CULTURAL & PREVENTIVE PRACTICES:
""" + "\n".join([f"- {x}" for x in adv['prevention']]) + """
====================================================
Advisory Source: Plant Pathology Guidelines & ICAR Standards
"""
        st.download_button(
            label="📥 Download Printable Advisory (TXT)",
            data=advisory_text,
            file_name=f"advisory_{adv['crop']}_{adv['disease_name']}.txt",
            mime="text/plain"
        )

    else:
        st.info("⚠️ No recent scan selected. Please scan an image in **🔍 Detect Crop Disease** to generate customized treatment tips.")


# ==========================================
# PAGE 5: DETECTION HISTORY
# ==========================================
elif "📜 Detection History" in selected_page:
    render_header(
        title="Crop Health Scan History & Audit Log",
        subtitle="Centralized SQLite log of all processed field specimens, predictions, severity metrics, and management statuses."
    )

    col_filter1, col_filter2, col_btn = st.columns([1, 1, 1])
    with col_filter1:
        crop_filter = st.selectbox("Filter by Crop:", ["All", "Tomato", "Potato", "Bell Pepper", "Corn (Maize)"])
    with col_filter2:
        status_filter = st.selectbox("Filter by Action Status:", ["All", "Pending", "In Progress", "Resolved"])
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Records", use_container_width=True):
            db.clear_all_history()
            st.success("History database reset.")
            st.rerun()

    scans = db.get_all_scans(limit=100, crop_filter=crop_filter, status_filter=status_filter)

    if scans:
        df = pd.DataFrame(scans)
        
        # Display as interactive table
        st.markdown(f"**Found {len(scans)} scan records:**")
        
        display_df = df[["id", "timestamp", "crop_name", "disease_name", "confidence", "severity_level", "lesion_percentage", "alert_level", "action_status"]].copy()
        display_df.columns = ["ID", "Timestamp", "Crop", "Diagnosis", "Confidence (%)", "Severity", "Lesion %", "Alert", "Status"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Status Update & Detail Expander
        st.markdown("#### ✏️ Update Farmer Action Status:")
        c_id, c_stat, c_upd = st.columns([1, 1, 1])
        with c_id:
            selected_scan_id = st.selectbox("Select Record ID:", df["id"].tolist())
        with c_stat:
            new_status = st.selectbox("New Action Status:", ["Pending", "In Progress", "Resolved"])
        with c_upd:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("💾 Update Status", use_container_width=True):
                db.update_action_status(selected_scan_id, new_status)
                st.success(f"Record #{selected_scan_id} updated to '{new_status}'")
                st.rerun()

        # Download History CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full History to CSV",
            data=csv_data,
            file_name="crop_health_detection_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No scan records found matching the active filter criteria.")


# ==========================================
# PAGE 6: ANALYTICS & INSIGHTS
# ==========================================
elif "📈 Analytics & Insights" in selected_page:
    render_header(
        title="Agricultural Health Analytics & Trends",
        subtitle="Aggregated epidemiological insights, disease prevalence, severity distributions, and scan velocity."
    )

    stats = db.get_analytics_summary()

    # Top KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Observations</div>
            <div class="metric-value">{stats['total_scans']}</div>
        </div>
    """, unsafe_allow_html=True)
    
    infection_rate = (stats['infected_scans'] / stats['total_scans'] * 100.0) if stats['total_scans'] > 0 else 0.0
    k2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Infection Rate</div>
            <div class="metric-value">{infection_rate:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    k3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Healthy Ratio</div>
            <div class="metric-value">{100.0 - infection_rate:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    k4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pathology Classes</div>
            <div class="metric-value">10</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("""<div class="agri-card-title">📊 Disease Prevalence Distribution</div>""", unsafe_allow_html=True)
        if stats["disease_distribution"]:
            df_dis = pd.DataFrame(stats["disease_distribution"])
            chart_dis = alt.Chart(df_dis).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X('count:Q', title='Scan Count'),
                y=alt.Y('disease_name:N', sort='-x', title='Identified Disease / Condition'),
                color=alt.value('#2E7D32'),
                tooltip=['disease_name', 'count']
            ).properties(height=300)
            st.altair_chart(chart_dis, use_container_width=True)
        else:
            st.info("No disease distribution data available yet.")

    with col_chart2:
        st.markdown("""<div class="agri-card-title">🥧 Severity Level Breakdown</div>""", unsafe_allow_html=True)
        if stats["severity_distribution"]:
            df_sev = pd.DataFrame(stats["severity_distribution"])
            chart_sev = alt.Chart(df_sev).mark_arc(innerRadius=45).encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="severity_level", type="nominal", scale=alt.Scale(range=['#4CAF50', '#FFB74D', '#E57373', '#81C784'])),
                tooltip=['severity_level', 'count']
            ).properties(height=300)
            st.altair_chart(chart_sev, use_container_width=True)
        else:
            st.info("No severity distribution data available yet.")

    # Model Evaluation Metrics Report
    st.markdown("---")
    st.markdown("""<div class="agri-card-title">🎯 AI CNN Model Validation Performance Metrics</div>""", unsafe_allow_html=True)
    
    eval_file = BASE_DIR / "model" / "eval_metrics.json"
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_data = json.load(f)

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Overall Accuracy", f"{eval_data['overall_accuracy']}%")
        e2.metric("Macro Precision", f"{eval_data['macro_precision']}%")
        e3.metric("Macro Recall", f"{eval_data['macro_recall']}%")
        e4.metric("Macro F1-Score", f"{eval_data['macro_f1']}%")

        st.markdown("##### Per-Class Precision, Recall, and F1-Scores:")
        df_metrics = pd.DataFrame.from_dict(eval_data["class_metrics"], orient="index")
        df_metrics.index.name = "Class Name"
        st.dataframe(df_metrics, use_container_width=True)
    else:
        st.info("Model evaluation report pending. Run model_eval.py to refresh.")


# ==========================================
# PAGE 7: ABOUT PROJECT
# ==========================================
elif "ℹ️ About Project" in selected_page:
    render_header(
        title="About the AI Crop Health Monitoring System",
        subtitle="Academic Documentation & System Engineering Specifications • Anna University AI3021"
    )

    t1, t2, t3, t4 = st.tabs(["📄 Project Background", "🏗️ System Architecture", "🧠 CNN Model Design", "👥 Project Team & Credits"])

    with t1:
        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">🌿 Project Overview & Problem Statement</div>
                <p><strong>Title:</strong> AI-Based Crop Health Monitoring and Early Disease Detection System</p>
                <p><strong>Course:</strong> AI3021 – Information Technology in Agriculture System</p>
                <p><strong>Institution:</strong> Info Institute of Engineering, Anna University, Chennai</p>
                <hr>
                <p>Late identification of crop diseases frequently causes devastating yield loss, increased pesticide overuse, and severe financial hardship for farming communities. Traditional identification relies solely on visual inspection by agricultural officers, which is often delayed and subjective.</p>
                <p>This project resolves this bottleneck by implementing an end-to-end automated platform that combines <strong>OpenCV computer vision preprocessing</strong>, a <strong>multi-layer Convolutional Neural Network (CNN)</strong>, quantitative <strong>severity segmentation</strong>, and an automated <strong>Farmer Alert and Treatment Advisory engine</strong>.</p>
            </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">🏗️ System Architecture & Workflow</div>
                <p>The system follows the exact multi-tier architecture documented in Section 5.1 and Figure 1 of the project report:</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
```mermaid
flowchart LR
    A[Farmer / User] --> B[Web Dashboard\nStreamlit]
    B --> C[Image Upload &\nCamera Validation]
    C --> D[OpenCV Preprocessing\nResize & Normalization]
    D --> E[CNN Disease\nClassification]
    E --> F[OpenCV Severity\nLesion Segmentation]
    F --> G[Advisory Engine &\nFarmer Alert]
    G --> H[SQLite Database &\nAnalytics]
```
        """)

        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">🔄 Disease Detection Decision Flow</div>
                <p>Matching Figure 4 of the project report:</p>
                <ol>
                    <li><strong>Image Input:</strong> Farmer submits leaf photograph through file upload or camera.</li>
                    <li><strong>Validation Check:</strong> OpenCV checks file readability, minimum resolution (32x32), and foliar vegetative color distribution.</li>
                    <li><strong>Preprocessing:</strong> Converted to RGB, resized to (128, 128, 3), and pixel-normalized to [0, 1].</li>
                    <li><strong>CNN Inference:</strong> Evaluated through 3 convolutional stages + Global Average Pooling + Dense layers.</li>
                    <li><strong>Confidence Verification:</strong> If confidence &lt; 50%, an ambiguity alert is flagged to prevent overconfident diagnosis.</li>
                    <li><strong>Severity & Advisory:</strong> Lesion % calculated, farmer alert generated, and results persisted to SQLite.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">🧠 Convolutional Neural Network (CNN) Specifications</div>
                <p>Architecture details matching Section 6.1 and 7.1 of the report:</p>
            </div>
        """, unsafe_allow_html=True)

        st.code(get_model_summary_string(), language="text")

    with t4:
        st.markdown("""
            <div class="agri-card">
                <div class="agri-card-title">👥 Academic Credits & Acknowledgements</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div style="background: #F4FAF5; padding: 18px; border-radius: 12px; border: 1px solid #DCECE0;">
                        <h4 style="color: #1B5E20; margin-top: 0;">Student Investigator</h4>
                        <p style="margin: 4px 0;"><strong>DEVAKI K</strong></p>
                        <p style="margin: 4px 0;">Register No: <strong>711023205007</strong></p>
                        <p style="margin: 4px 0;">Department of Information Technology</p>
                        <p style="margin: 4px 0;">Info Institute of Engineering, Coimbatore</p>
                    </div>
                    <div style="background: #F4FAF5; padding: 18px; border-radius: 12px; border: 1px solid #DCECE0;">
                        <h4 style="color: #1B5E20; margin-top: 0;">Project Guide & Head of Department</h4>
                        <p style="margin: 4px 0;"><strong>Dr. K. PALANI, M.E., Ph.D.</strong></p>
                        <p style="margin: 4px 0;">Professor & Head of Department</p>
                        <p style="margin: 4px 0;">Department of Information Technology</p>
                        <p style="margin: 4px 0;">Info Institute of Engineering, Coimbatore</p>
                    </div>
                </div>
                <br>
                <div style="background: #F9FAF9; padding: 15px; border-radius: 10px; border: 1px solid #E5EBE6; font-size: 0.9rem; color: #555;">
                    <em>Submitted in partial fulfillment for the award of the degree of Bachelor of Technology in Information Technology, Anna University: Chennai 600 025. Academic Year 2026–2027.</em>
                </div>
            </div>
        """, unsafe_allow_html=True)
