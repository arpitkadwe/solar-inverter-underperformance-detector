@echo off
title Solar Inverter Underperformance Detector
cd /d "%~dp0"
python -m streamlit run app\app.py
pause