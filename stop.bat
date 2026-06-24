@echo off
title Qwen3.5-4B - Stop Services

cd /d "%~dp0"

echo.
echo ========================================
echo   Stopping all Qwen3.5-4B Services
echo ========================================
echo.

echo Stopping all containers and cleaning up...
docker compose down --remove-orphans
docker container prune -f --filter "label=com.docker.compose.project=qwen"

echo.
echo All services stopped.
echo.
pause
