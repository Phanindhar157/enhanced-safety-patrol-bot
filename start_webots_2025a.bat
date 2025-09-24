@echo off
echo 🤖 Enhanced Safety Patrol Bot - Webots 2025a Launcher
echo ============================================================
echo.
echo Starting your enhanced safety patrol bot system...
echo.

echo 📊 Starting Web Dashboard...
start python start_dashboard.py
timeout /t 2 /nobreak >nul

echo 🌐 Opening Web Dashboard...
start http://localhost:8000/dashboard.html

echo.
echo 🎯 Next Steps:
echo 1. Open Webots 2025a
echo 2. Load: enhanced_safety_patrol_bot.wbt
echo 3. Set controller: enhanced_patrol_controller
echo 4. Click Play to start simulation
echo.
echo 📱 Monitor via: http://localhost:8000/dashboard.html
echo.
echo ✅ System ready for Webots 2025a!
pause

