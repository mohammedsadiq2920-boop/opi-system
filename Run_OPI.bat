@echo off
:: الدخول لمجلد المشروع لضمان عمل قاعدة البيانات
cd /d "%~dp0"
title OPI Production System Launcher

echo ========================================
echo   OPI - Oil Production Improvement
echo ========================================
echo.

:: 1. فتح المتصفح
start http://127.0.0.1:5000

:: 2. تشغيل الملف البرمجي الأساسي (app.py)
python app.py

pause