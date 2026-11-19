"""
UI Styling and Custom HTML/CSS Component Library
Modern Green and White Agricultural Theme
AI-Based Crop Health Monitoring and Early Disease Detection System
"""

import streamlit as st

def apply_agricultural_theme():
    """Injects custom CSS styling for a modern green-and-white agricultural theme."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1F2E24;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F5FAF6 0%, #FFFFFF 100%);
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Header & Branding */
    .agri-header {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(27, 94, 32, 0.15);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .agri-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.3rem;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
    }
    .agri-header p {
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        color: #E8F5E9 !important;
        font-weight: 400;
    }
    .agri-header .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.35rem 0.85rem;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        backdrop-filter: blur(4px);
    }

    /* Agricultural Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E1EBE2;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.1);
        border-color: #81C784;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2E7D32;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #55755E;
        margin-top: 0.3rem;
    }

    /* Card Panels */
    .agri-card {
        background: #FFFFFF;
        border: 1px solid #E3ECE4;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 6px 18px rgba(35, 75, 45, 0.04);
        margin-bottom: 1.5rem;
    }
    .agri-card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B5E20;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E8F5E9;
        padding-bottom: 0.6rem;
    }

    /* Alert Banners */
    .alert-banner-emergency {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 6px solid #D32F2F;
        color: #B71C1C;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        box-shadow: 0 4px 12px rgba(211, 47, 47, 0.08);
    }
    .alert-banner-warning {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 6px solid #F57C00;
        color: #E65100;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        box-shadow: 0 4px 12px rgba(245, 124, 0, 0.08);
    }
    .alert-banner-safe {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #2E7D32;
        color: #1B5E20;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.08);
    }

    /* Recommendation Badges & List */
    .treatment-item {
        background: #F8FAF8;
        border-left: 4px solid #4CAF50;
        padding: 0.85rem 1.1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: #26382B;
    }
    .treatment-item strong {
        color: #1B5E20;
    }

    /* Streamlit Button Restyling */
    .stButton>button {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #388E3C 0%, #2E7D32 100%);
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.3);
        transform: translateY(-1px);
        color: white;
    }

    /* File Uploader Box */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 2px dashed #81C784;
        border-radius: 16px;
        padding: 1rem;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #2E7D32;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F1F6F2 !important;
        border-right: 1px solid #E0E8E1;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.4rem;
        color: #1B5E20;
    }

    /* Severity Indicator Bar */
    .severity-badge-mild {
        background-color: #FFF9C4;
        color: #F57F17;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
    }
    .severity-badge-moderate {
        background-color: #FFE0B2;
        color: #E65100;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
    }
    .severity-badge-severe {
        background-color: #FFCDD2;
        color: #C62828;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
    }
    .severity-badge-healthy {
        background-color: #C8E6C9;
        color: #1B5E20;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_header(title="Crop Health Monitoring System", subtitle="AI-Powered Early Disease Detection & Agricultural Decision Support"):
    """Renders the top banner matching modern agricultural software."""
    html = f"""
    <div class="agri-header">
        <span class="badge">🌱 AI-DRIVEN AGRI-TECH PLATFORM • ANNA UNIVERSITY AI3021</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_metric_card(label, value, icon="🌿", delta=None):
    """Renders a styled metric card."""
    delta_html = f"<div style='font-size: 0.8rem; color: #388E3C; margin-top: 4px;'>{delta}</div>" if delta else ""
    return f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="metric-label">{label}</div>
            <div style="font-size: 1.5rem;">{icon}</div>
        </div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def render_alert_banner(alert_level, badge, message):
    """Renders styled farmer alerts with high visibility."""
    css_class = "alert-banner-safe"
    if alert_level == "Emergency":
        css_class = "alert-banner-emergency"
    elif alert_level == "Warning":
        css_class = "alert-banner-warning"

    html = f"""
    <div class="{css_class}">
        <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.5rem;">
            {badge}
        </div>
        <div style="font-size: 0.98rem; line-height: 1.45;">
            {message}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
