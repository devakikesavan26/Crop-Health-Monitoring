@echo off
echo ===================================================
echo Starting AI-Based Crop Health Monitoring System...
echo ===================================================
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501
pause
