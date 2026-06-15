@echo off
title BVL Quant Engine - Dashboard
color 0B

echo.
echo  =============================================
echo   BVL QUANT ENGINE - Institutional Dashboard
echo  =============================================
echo.
echo  Iniciando servidor...
echo  El dashboard estara disponible en:
echo.
echo      http://127.0.0.1:8000
echo.
echo  Abre tu navegador en esa direccion.
echo  Para detener el servidor presiona Ctrl+C
echo.
echo  =============================================
echo.

cd /d "%~dp0"
python api.py

pause
