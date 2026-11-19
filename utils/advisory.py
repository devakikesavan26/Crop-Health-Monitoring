"""
Agricultural Advisory and Farmer Alert Module
AI-Based Crop Health Monitoring and Early Disease Detection System
"""

from config import DISEASE_INFO

class AdvisoryEngine:
    @staticmethod
    def get_advisory(class_name, confidence=100.0, severity_info=None):
        """
        Retrieves comprehensive agricultural advisory, alert banners, and treatment recommendations.
        """
        info = DISEASE_INFO.get(class_name)
        if not info:
            # Fallback for unknown classes
            return {
                "crop": "Unspecified Crop",
                "disease_name": class_name.replace("___", " ").replace("_", " "),
                "status": "Unknown",
                "pathogen": "Unknown Pathogen",
                "symptoms": "Symptoms not cataloged in current database.",
                "alert_level": "Warning",
                "alert_badge": "⚠️ Unverified Detection",
                "alert_message": "Low certainty or unverified disease pattern. Please consult a local agricultural extension officer.",
                "organic_treatment": ["Monitor plant closely and isolate if symptoms worsen."],
                "chemical_treatment": ["Do not apply unverified chemical fungicides."],
                "prevention": ["Ensure balanced irrigation and avoid overhead wetting."]
            }

        # Dynamic alert severity enhancement
        alert_level = info["alert_level"]
        alert_badge = info["alert_badge"]
        alert_message = info["alert_message"]

        if severity_info:
            lesion_pct = severity_info.get("lesion_percentage", 0.0)
            if lesion_pct > 30.0 and info["status"] != "Healthy":
                alert_level = "Emergency"
                alert_badge = "🚨 CRITICAL OUTBREAK ALERT"
                alert_message = f"HIGH SEVERITY DETECTED ({lesion_pct}% leaf area affected). High risk of field-wide crop loss! Immediate curative fungicidal treatment required."

        # Low confidence warning check
        confidence_warning = None
        if confidence < 50.0:
            confidence_warning = (
                "⚠️ Prediction confidence is below 50%. The visual pattern may be ambiguous. "
                "Verify with physical leaf examination or take a clearer photograph under direct natural lighting."
            )

        return {
            "crop": info["crop"],
            "disease_name": info["disease_name"],
            "status": info["status"],
            "pathogen": info["pathogen"],
            "symptoms": info["symptoms"],
            "alert_level": alert_level,
            "alert_badge": alert_badge,
            "alert_message": alert_message,
            "confidence_warning": confidence_warning,
            "organic_treatment": info["organic_treatment"],
            "chemical_treatment": info["chemical_treatment"],
            "prevention": info["prevention"]
        }
