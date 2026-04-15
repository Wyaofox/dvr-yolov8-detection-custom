@echo off
title YOLO Surveillance System
cd /d E:\jiankong_project\dvr-yolov8-detection
echo ========================================
echo   YOLO Surveillance - Starting
echo ========================================
echo.
echo Web:  http://127.0.0.1:5000
echo NiceGUI: http://127.0.0.1:5001
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.
C:\Users\W_yao\AppData\Local\Programs\Python\Python313\python.exe -u yolov8_live_rtmp_stream_detection.py
pause
