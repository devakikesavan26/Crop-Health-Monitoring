"""
Configuration and Agricultural Disease Knowledge Base
AI-Based Crop Health Monitoring and Early Disease Detection System
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "data" / "crop_health.db"
MODEL_DIR = BASE_DIR / "model"
CNN_WEIGHTS_PATH = MODEL_DIR / "crop_disease_cnn.npz"
KERAS_MODEL_PATH = MODEL_DIR / "disease_detection_model.h5"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Image settings
IMAGE_SIZE = (128, 128)
INPUT_CHANNELS = 3

# Supported Disease Classes
CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper__bell___healthy",
    "Pepper__bell___Bacterial_spot",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_"
]

# Detailed Agricultural Metadata for Each Class
DISEASE_INFO = {
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease_name": "Healthy Leaf",
        "status": "Healthy",
        "pathogen": "None (Optimum Plant Health)",
        "symptoms": "Leaf tissue is uniformly green with smooth margins. No visible chlorosis, necrosis, leaf spots, or fungal mycelium.",
        "severity_default": "None (0%)",
        "alert_level": "Safe",
        "alert_badge": "🟢 All Clear",
        "alert_message": "Your tomato plant shows strong vigor and healthy foliar growth. No active infection detected.",
        "organic_treatment": [
            "Maintain regular soil moisture using drip irrigation.",
            "Apply prophylactic neem oil spray (3 ml/L) once every 14 days to deter sap-sucking pests.",
            "Mulch the soil bed with clean straw to prevent soil-splash pathogens."
        ],
        "chemical_treatment": [
            "No chemical fungicides or bactericides required.",
            "Apply balanced NPK (19:19:19) foliar spray at 4-5 g/L if vegetative boost is needed."
        ],
        "prevention": [
            "Ensure 45-60 cm spacing between rows for optimal sunlight and air circulation.",
            "Avoid overhead watering during late evenings.",
            "Scout weekly for early signs of yellowing or flea beetles."
        ]
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease_name": "Early Blight",
        "status": "Infected",
        "pathogen": "Alternaria solani (Fungal)",
        "symptoms": "Small brown-black spots with concentric rings ('target-board' appearance) on older lower leaves. Chlorotic yellow halos surround spots, leading to defoliation.",
        "severity_default": "Moderate",
        "alert_level": "Warning",
        "alert_badge": "🟠 Active Fungal Threat",
        "alert_message": "Early Blight detected. Spores can spread rapidly to healthy upper leaves in humid conditions (>80% RH). Immediate action recommended.",
        "organic_treatment": [
            "Spray Copper Hydroxide or liquid copper fungicide at 2.5 g/L of water.",
            "Apply Trichoderma viride or Bacillus subtilis bio-fungicide (5 g/L) to soil and foliar canopy.",
            "Prune infected lower foliage immediately and burn/dispose far away from field."
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP @ 2.0-2.5 g/L water (protective spray).",
            "Chlorothalonil 75% WP @ 2 g/L or Azoxystrobin 23% SC @ 1 ml/L (systemic curative).",
            "Repeat spray at 7-10 day intervals during wet weather."
        ],
        "prevention": [
            "Implement 3-year crop rotation avoiding Solanaceae family (potatoes, peppers, eggplants).",
            "Stake plants off the ground and prune bottom leaves 20 cm above ground.",
            "Disinfect pruning shears with 70% isopropyl alcohol between plants."
        ]
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease_name": "Late Blight",
        "status": "Infected",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Large water-soaked irregular dark olive to purplish lesions that expand rapidly. White cottony fungal sporulation appears on leaf undersides in humid conditions.",
        "severity_default": "High",
        "alert_level": "Emergency",
        "alert_badge": "🔴 Critical Crop Alert",
        "alert_message": "CRITICAL: Late Blight is highly destructive and can ruin an entire tomato plot in 7-10 days under cool, wet conditions (15-22°C). Act within 24 hours!",
        "organic_treatment": [
            "Spray Bordeaux mixture (1% w/v: copper sulfate + slaked lime) immediately.",
            "Foliar spray of Potassium Phosphite (3 ml/L) to trigger systemic acquired resistance.",
            "Remove and incinerate heavily infected plants. Do not compost infected leaves."
        ],
        "chemical_treatment": [
            "Metalaxyl-M 4% + Mancozeb 64% WP (Ridomil Gold) @ 2.5 g/L water immediately.",
            "Cymoxanil 8% + Mancozeb 64% WP (Curzate) @ 2.5 g/L as curative intervention.",
            "Dimethomorph 50% WP @ 1 g/L in rotational schedule to prevent resistance."
        ],
        "prevention": [
            "Plant certified disease-resistant tomato hybrids.",
            "Eliminate all volunteer tomato and potato plants from previous season.",
            "Maintain wide ridge spacing and ensure zero standing water in the field."
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease_name": "Healthy Leaf",
        "status": "Healthy",
        "pathogen": "None (Normal Foliage)",
        "symptoms": "Vibrant compound foliage with intact margins, uniform deep green pigmentation, and clean stem junctions.",
        "severity_default": "None (0%)",
        "alert_level": "Safe",
        "alert_badge": "🟢 All Clear",
        "alert_message": "Potato foliage is healthy. Strong photosynthetic capacity supports optimum tuber bulking.",
        "organic_treatment": [
            "Maintain soil hilling around potato ridges to prevent tuber greening.",
            "Apply prophylactic bio-agents (Trichoderma harzianum) in irrigation channels.",
            "Spray seaweed extract (2 ml/L) for micronutrient supply and plant immunity."
        ],
        "chemical_treatment": [
            "No chemical fungicides needed.",
            "Maintain recommended soil potassium (Potash) for sturdy stem walls."
        ],
        "prevention": [
            "Inspect weekly after rainy periods or heavy early-morning dew.",
            "Keep potato fields free from weeds and nightshade relatives."
        ]
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease_name": "Early Blight",
        "status": "Infected",
        "pathogen": "Alternaria solani (Fungal)",
        "symptoms": "Dark brown necrotic spots with concentric ring patterns on older leaflets. Stems may show sunken elongated lesions.",
        "severity_default": "Moderate",
        "alert_level": "Warning",
        "alert_badge": "🟠 Active Fungal Threat",
        "alert_message": "Early Blight detected on potato foliage. Reduces leaf surface area and diminishes tuber yield if not contained.",
        "organic_treatment": [
            "Copper oxychloride 50% WP @ 3 g/L as preventive barrier.",
            "Neem cake extract (5%) soil incorporation to reduce overwintering conidia.",
            "Remove dried-up lower infected leaflets."
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP @ 2.5 g/L water as initial protective spray.",
            "Propiconazole 25% EC @ 1 ml/L or Difenoconazole 25% EC @ 0.5 ml/L for curative action.",
            "Apply early morning before wind speeds pick up."
        ],
        "prevention": [
            "Avoid nitrogen stress; under-fertilized potato crops succumb quicker to Alternaria.",
            "Ensure 30-40 cm furrow drainage to prevent humidity pockets.",
            "Do not follow potatoes after tomatoes in crop rotations."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease_name": "Late Blight",
        "status": "Infected",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Water-soaked dark lesions spreading rapidly across leaflet tips and petioles. Under humid conditions, a white velvety mold forms underneath.",
        "severity_default": "High",
        "alert_level": "Emergency",
        "alert_badge": "🔴 Critical Crop Alert",
        "alert_message": "EMERGENCY: Late Blight causes catastrophic tuber rot and vine collapse. Fungicidal spray must be initiated immediately across the field.",
        "organic_treatment": [
            "Bordeaux mixture (1%) or Copper hydroxide 53.8% DF @ 2 g/L.",
            "Cut vines (dehaulming) 10-14 days prior to harvest if infection reaches late stage, to protect tubers.",
            "Solarize or burn destroyed plant canopy."
        ],
        "chemical_treatment": [
            "Mandipropamid 23.4% SC @ 0.8 ml/L or Metalaxyl + Mancozeb @ 2.5 g/L.",
            "Fluopicolide + Propamocarb hydrochloride (Infinito) @ 2.5 ml/L.",
            "Alternate fungicides with different modes of action to avert strain resistance."
        ],
        "prevention": [
            "Use certified disease-free seed tubers from verified agricultural departments.",
            "Avoid night overhead sprinkler irrigation.",
            "Earth up rows well to create a 10 cm soil barrier protecting developing tubers from washed-off spores."
        ]
    },
    "Pepper__bell___healthy": {
        "crop": "Bell Pepper / Capsicum",
        "disease_name": "Healthy Leaf",
        "status": "Healthy",
        "pathogen": "None (Normal Foliage)",
        "symptoms": "Firm, glossy green leaves without chlorosis, mottling, or necrotic spots. Normal venation and leaf turgor.",
        "severity_default": "None (0%)",
        "alert_level": "Safe",
        "alert_badge": "🟢 All Clear",
        "alert_message": "Bell pepper canopy is healthy and active. Ideal for flower set and fruit development.",
        "organic_treatment": [
            "Spray Panchagavya (3%) or vermiwash (5%) every 15 days for robust vegetative growth.",
            "Maintain mulch cover to regulate root temperature."
        ],
        "chemical_treatment": [
            "No chemical fungicides or bactericides required.",
            "Foliar calcium nitrate spray @ 2 g/L to prevent blossom end rot in developing fruits."
        ],
        "prevention": [
            "Scout for thrips and mites on undersides of tender top leaves.",
            "Ensure soil has adequate organic matter and pH between 6.0 and 6.8."
        ]
    },
    "Pepper__bell___Bacterial_spot": {
        "crop": "Bell Pepper / Capsicum",
        "disease_name": "Bacterial Spot",
        "status": "Infected",
        "pathogen": "Xanthomonas campestris pv. vesicatoria (Bacterial)",
        "symptoms": "Small, water-soaked, irregular circular spots turning dark brown to black with yellow halos. Spots often coalesce, causing severe leaf drop and fruit lesions.",
        "severity_default": "High",
        "alert_level": "Emergency",
        "alert_badge": "🔴 Bacterial Threat",
        "alert_message": "Bacterial Spot spreads through rain splashing and handling wet plants. Copper + Bactericide application is urgent.",
        "organic_treatment": [
            "Copper hydroxide + Streptomycin sulfate alternative spray.",
            "Bacillus amyloliquefaciens bio-bactericide foliar application.",
            "Immediately avoid working in pepper fields while foliage is wet."
        ],
        "chemical_treatment": [
            "Copper Oxychloride 50% WP @ 2.5 g/L + Streptocycline (Plantomycin) @ 1 g per 10 L water.",
            "Kasugamycin 3% SL @ 2 ml/L water.",
            "Repeat treatment after 7 days if wet weather persists."
        ],
        "prevention": [
            "Plant hot water treated or certified disease-free seeds.",
            "Use drip irrigation exclusively; never use overhead sprinklers.",
            "Rotate out of pepper/tomato crops for at least 2 seasons."
        ]
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "disease_name": "Healthy Leaf",
        "status": "Healthy",
        "pathogen": "None (Optimal Growth)",
        "symptoms": "Elongated, deep green leaf blades with clean midribs and no rust pustules, blights, or insect bore holes.",
        "severity_default": "None (0%)",
        "alert_level": "Safe",
        "alert_badge": "🟢 All Clear",
        "alert_message": "Maize crop exhibits healthy photosynthetic surface area. High yield potential expected.",
        "organic_treatment": [
            "Ensure regular irrigation during critical flowering (tasseling and silking) stages.",
            "Apply Azospirillum and Phosphobacteria bio-fertilizers."
        ],
        "chemical_treatment": [
            "No chemical treatment required.",
            "Side-dress urea or balanced nitrogen according to soil test recommendations."
        ],
        "prevention": [
            "Maintain plant density of 60 cm x 20 cm.",
            "Monitor weekly for fall armyworm or stem borer activity."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "disease_name": "Common Rust",
        "status": "Infected",
        "pathogen": "Puccinia sorghi (Fungal)",
        "symptoms": "Oval to elongated cinnamon-brown to golden-brown powdery pustules (uredinia) scattered on both upper and lower leaf surfaces. Pustules rupture epidermal tissue releasing rust spores.",
        "severity_default": "Moderate",
        "alert_level": "Warning",
        "alert_badge": "🟠 Fungal Rust Warning",
        "alert_message": "Common Rust detected on maize leaves. Windblown spores can spread swiftly during cool, humid weather (16-25°C).",
        "organic_treatment": [
            "Wettable sulfur 80% WP @ 3 g/L as organic anti-fungal barrier.",
            "Neem oil 1% spray to suppress fungal spore germination.",
            "Improve field air drainage to accelerate morning leaf drying."
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP @ 2.5 g/L water at initial pustule emergence.",
            "Azoxystrobin 18.2% + Difenoconazole 11.4% SC (Amistar Top) @ 1 ml/L.",
            "Propiconazole 25% EC (Tilt) @ 1 ml/L if rust reaches ear leaves during tasseling."
        ],
        "prevention": [
            "Plant rust-resistant or tolerant corn hybrids.",
            "Plant early in the season to evade late-season peak spore flights.",
            "Eradicate Oxalis weeds (wood sorrel) nearby as they serve as alternate hosts."
        ]
    }
}

# UI Colors (Modern Green & White Theme)
THEME_COLORS = {
    "primary": "#2E7D32",       # Forest green
    "primary_dark": "#1B5E20",  # Dark green
    "primary_light": "#4CAF50", # Leaf green
    "secondary": "#81C784",     # Soft light green
    "accent": "#A5D6A7",        # Pale green accent
    "background": "#F7FAF7",    # Off-white agricultural background
    "card_bg": "#FFFFFF",       # Pure white cards
    "text": "#1C2826",          # Deep slate text
    "text_muted": "#556B2F",    # Olive muted text
    "danger": "#D32F2F",        # Red alert
    "warning": "#F57C00",       # Orange alert
    "success": "#388E3C",       # Success green
    "border": "#E0E8E0"         # Soft green-tinted border
}
