@echo off
setlocal enabledelayedexpansion
title Qwen3.5-4B 高并发 OCR + JSON 抽取处理管线

cd /d "%~dp0"

echo.
echo =========================================================
echo   Qwen3.5-4B 本地处理管线 (OCR + 结构化抽取)
echo   模式: 高并发分组处理
echo =========================================================
echo.

:: ----------------------------------------------------------
:: Step 1: 检查运行环境
:: ----------------------------------------------------------
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop 未运行！请先启动 Docker。
    pause
    exit /b 1
)
echo [1/4] Docker Desktop 已就绪。

:: ----------------------------------------------------------
:: Step 2: 启动 vLLM 引擎端
:: ----------------------------------------------------------
echo [2/4] 正在启动 vLLM 大模型推理引擎...

docker compose up -d vllm-server >nul 2>&1

:: 弹出一个独立窗口，专门用来实时滚动查看 vLLM 的抢占警告和底层日志
start cmd /k "title vLLM Server Logs && echo 正在实时追踪 vLLM 引擎后台日志... && echo 如遇 [Preemption] 抢占警告，请到 .env 调低 MAX_NUM_SEQS && echo ================================================================ && docker logs -f qwen-vllm-server"

echo       vLLM 引擎已在后台启动。
echo       (已为您弹出独立日志窗口，可以实时观察大模型的显存占用和加载进度)
echo.

:: ----------------------------------------------------------
:: Step 3: 等待大模型完全载入显存
:: ----------------------------------------------------------
echo [3/4] 正在等待模型载入显存 (首次约需要一两分钟)...
:wait_loop
curl -sf http://127.0.0.1:8000/v1/models >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo.
    echo       ✅ vLLM 引擎就绪！模型已完全载入。
    echo.
    goto :start_batch
)
timeout /t 3 >nul
goto :wait_loop

:: ----------------------------------------------------------
:: Step 4: 启动高并发处理客户端
:: ----------------------------------------------------------
:start_batch
echo [4/4] 启动客户端批处理任务...
echo.
echo =========================================================
echo   系统就绪！将按文件夹分组并发处理 input\ 目录下的图像
echo   处理流程：
echo     1. 读取配置 (OCR温度/并发数)
echo     2. 启动并发线程池进行 OCR 文字识别
echo     3. 按病人分组生成 merged_ocr.md
echo     4. 自动调用 Qwen 对长文本进行 JSON 结构化抽取
echo =========================================================
echo.

:: 调用 python process.py
docker compose run --rm qwen-client python /workspace/scripts/process.py

echo.
echo =========================================================
echo   批处理任务已全部结束！
echo   请前往 output\ 目录查看结果。
echo =========================================================
echo.

echo (后台推理引擎将继续保持运行，若想关闭请运行 stop.bat)
echo.
pause
