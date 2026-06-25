#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR-VL-1.6 批处理脚本 (PPLayout 版面分析 + vLLM 推理后端)
功能:
  - 扫描 input/ 目录中的图片和 PDF 文件（按病人分组）
  - 使用 PaddleOCR SDK (PPLayout 版面分析 + vLLM Server VLM 推理后端) 进行 OCR 识别
  - PPLayout 自动完成：版面检测 → 区域裁切 → 子图 OCR → Markdown 拼接
  - 合并所有文件的 OCR 结果
  - 针对合并后的长文本，统一进行一次全局 JSON 结构化抽取（可选）
  - 处理完的文件移到 input/processed/
"""

import os
import sys
import json
import shutil
import time
import base64
import traceback
import re
from pathlib import Path
from datetime import datetime

import yaml
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# =============================================================================
# 全局时间戳日志拦截器
# =============================================================================
import builtins
import threading

_original_print = builtins.print
_print_lock = threading.Lock()

def timestamped_print(*args, **kwargs):
    from datetime import datetime, timezone, timedelta
    tz_shanghai = timezone(timedelta(hours=8))
    time_str = datetime.now(tz_shanghai).strftime("[%H:%M:%S]")
    with _print_lock:
        if args and isinstance(args[0], str) and args[0].startswith("\n"):
            # 如果原先是换行打头，则把换行提到时间戳外面，保持版式美观
            first_arg = args[0][1:]
            if first_arg:
                _original_print("\n" + time_str + " " + first_arg, *args[1:], **kwargs)
            else:
                _original_print("\n" + time_str, *args[1:], **kwargs)
        else:
            _original_print(time_str, *args, **kwargs)

builtins.print = timestamped_print



# =============================================================================
# 配置加载
# =============================================================================

def load_config(config_path: str = "/workspace/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# vLLM Client 初始化
# =============================================================================

def create_client(server_url: str) -> OpenAI:
    """创建 OpenAI 兼容客户端 (连接 vLLM Server)"""
    print(f"[INFO] 连接 vLLM Server: {server_url}")
    client = OpenAI(
        base_url=server_url,
        api_key="not-needed",  # vLLM 不需要 API key
    )
    return client


def get_model_name(client: OpenAI) -> str:
    """从 vLLM Server 获取加载的模型名称"""
    models = client.models.list()
    model_name = models.data[0].id
    print(f"[INFO] 模型: {model_name}")
    return model_name


# =============================================================================
# 图片编码
# =============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """获取图片 MIME 类型"""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")


# =============================================================================
# PDF 处理
# =============================================================================

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    """将 PDF 转换为图片列表"""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []
        temp_dir = Path(pdf_path).parent / f".tmp_{Path(pdf_path).stem}"
        temp_dir.mkdir(exist_ok=True)

        for i, img in enumerate(images):
            img_path = temp_dir / f"page_{i+1:03d}.png"
            img.save(str(img_path), "PNG")
            image_paths.append(str(img_path))

        return image_paths
    except Exception as e:
        print(f"[ERROR] PDF 转换失败: {e}")
        return []


# =============================================================================
# OCR 识别 (阶段一)
# =============================================================================

def ocr_image(
    client: OpenAI,
    model_name: str,
    image_path: str,
    config: dict,
) -> str:
    """使用 Qwen3.5-4B 视觉能力识别图片中的文字"""
    ocr_config = config.get("ocr", {})
    inference = config.get("inference", {})

    system_prompt = ocr_config.get("system_prompt", "请识别图片中的所有文字内容。")
    user_prompt = ocr_config.get("user_prompt", "请识别这张图片中的所有文字内容。")

    # 编码图片
    base64_image = encode_image_to_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]

    # 是否启用原生思考模式
    thinking_enabled = inference.get("thinking_enabled", False)
    extra_body = {
        "top_k": inference.get("top_k", 20),
        "min_p": inference.get("min_p", 0.0),
        "repetition_penalty": inference.get("repetition_penalty", 1.0),
    }
    if not thinking_enabled:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=inference.get("temperature", 0.0),
        max_tokens=inference.get("max_tokens", 4096),
        top_p=inference.get("top_p", 0.95),
        presence_penalty=inference.get("presence_penalty", 0.0),
        extra_body=extra_body,
    )

    return response.choices[0].message.content


def build_schema_template(extraction: dict) -> str:
    """构建压缩版中文抽取模板。

    从 config.yaml 的 extraction.schema_template 读取模板。
    模板中的 T/J 是给模型看的压缩字段类型。
    """
    schema_template = extraction.get("schema_template")
    if schema_template is None:
        raise ValueError("缺少 extraction.schema_template 配置")

    return json.dumps(schema_template, ensure_ascii=False, indent=2)


def restore_evidence(position, anchor_map: dict) -> str | None:
    """把模型输出的锚点标签（如 <s1> 或 ["<s1>", "<s2>"]）还原为原文证据片段。"""
    if not position:
        return None
    
    if isinstance(position, str):
        return anchor_map.get(position)
        
    if isinstance(position, list):
        evidences = [anchor_map.get(str(p)) for p in position if str(p) in anchor_map]
        return "".join(evidences) if evidences else None
        
    return None


def restore_compact_output(data, anchor_map: dict):
    """把模型输出的压缩格式还原为中文结构化 JSON。"""
    status_map = {
        0: "正常",
        1: "异常",
        2: "未提及",
        "0": "正常",
        "1": "异常",
        "2": "未提及",
    }

    if isinstance(data, dict):
        keys = set(data.keys())

        # T 类型叶子节点：原文截取型
        if keys.issubset({"v", "p"}) and "v" in data:
            return {
                "值": data.get("v"),
                "证据": restore_evidence(data.get("p"), anchor_map),
            }

        # J 类型叶子节点：异常判断型
        if keys.issubset({"s", "p"}) and "s" in data:
            return {
                "状态": status_map.get(data.get("s"), "未提及"),
                "异常证据": restore_evidence(data.get("p"), anchor_map),
            }

        # 普通嵌套对象
        return {
            key: restore_compact_output(value, anchor_map)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            restore_compact_output(item, anchor_map)
            for item in data
        ]

    return data


# =============================================================================
# 结构化提取 (阶段二)
# =============================================================================

def extract_structured(
    client: OpenAI,
    model_name: str,
    ocr_text: str,
    config: dict,
) -> dict:
    """从文字中提取结构化 JSON"""
    extraction = config.get("extraction", {})
    inference = config.get("inference", {})

    if not extraction.get("enabled", True):
        return None

    system_prompt = extraction.get("system_prompt", "从文本中提取结构化信息。")
    user_prompt_template = extraction.get("user_prompt_template", "请提取信息：\n{text}")

    schema_template = build_schema_template(extraction)

    # 注入隐形锚点标记
    import re
    parts = re.split(r'([。！？\n，,；;]+)', ocr_text)
    
    anchored_text = ""
    anchor_map = {}
    anchor_idx = 1
    current_chunk = ""
    
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        punct = parts[i+1] if i+1 < len(parts) else ""
        
        current_chunk += chunk + punct
        
        if current_chunk.strip():
            tag = f"<s{anchor_idx}>"
            anchored_text += current_chunk + tag
            # 去除可能的前后空格或换行作为纯净的证据
            anchor_map[tag] = current_chunk.strip()
            anchor_idx += 1
            current_chunk = ""
        else:
            anchored_text += current_chunk
            current_chunk = ""

    numbered_text = anchored_text

    # 格式化 user prompt
    user_prompt = user_prompt_template.format(
        text=numbered_text,
        schema_template=schema_template,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 是否启用原生思考模式
    thinking_enabled = inference.get("thinking_enabled", False)
    extra_body = {
        "top_k": inference.get("top_k", 20),
        "min_p": inference.get("min_p", 0.0),
        "repetition_penalty": inference.get("repetition_penalty", 1.0),
    }
    if not thinking_enabled:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=inference.get("temperature", 0.0),
        max_tokens=inference.get("max_tokens", 4096),
        top_p=inference.get("top_p", 0.95),
        presence_penalty=inference.get("presence_penalty", 0.0),
        extra_body=extra_body,
        stream=False,
    )

    raw_response = response.choices[0].message.content

    # 尝试解析 JSON
    try:
        # 处理可能被 markdown 代码块包裹的 JSON
        json_str = raw_response.strip()
        # 强力清理可能产生的思考过程 (处理只有关闭标签的情况)
        if "</think>" in json_str:
            json_str = json_str.split("</think>")[-1].strip()
        else:
            json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL).strip()
            
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        parsed_json = json.loads(json_str)

        parsed_json = restore_compact_output(parsed_json, anchor_map)
        return parsed_json
    except json.JSONDecodeError:
        print(f"[WARN] JSON 解析失败, 返回原始文本")
        return {"_raw_response": raw_response, "_parse_error": True}


# =============================================================================
# 文件扫描
# =============================================================================

def scan_input_dir(input_dir: str, supported_ext: list) -> dict:
    """扫描输入目录，返回按目录分组的待处理文件列表"""
    grouped_files = {}
    input_path = Path(input_dir)

    for item in sorted(input_path.iterdir()):
        if item.name.startswith(".") or item.name == "processed":
            continue
            
        if item.is_dir():
            group_name = item.name
            files = []
            for subitem in sorted(item.iterdir()):
                if subitem.is_dir() or subitem.name.startswith("."):
                    continue
                ext = subitem.suffix.lower()
                if ext in supported_ext:
                    files.append(subitem)
                else:
                    print(f"[WARN] 跳过不支持的文件: {group_name}/{subitem.name}")
            if files:
                grouped_files[group_name] = files
        else:
            ext = item.suffix.lower()
            if ext in supported_ext:
                if "_ungrouped" not in grouped_files:
                    grouped_files["_ungrouped"] = []
                grouped_files["_ungrouped"].append(item)
            else:
                print(f"[WARN] 跳过不支持的文件: {item.name}")

    return grouped_files



# =============================================================================
# 单文件处理 (Qwen-VL Pipeline)
# =============================================================================

def process_single_file_qwen(
    client,
    model_name: str,
    file_path: Path,
    output_dir: Path,
    config: dict
) -> dict:
    """使用 Qwen-VL 处理单个文件 (直接 OCR)"""
    file_name = file_path.stem
    file_output_dir = output_dir / file_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        ocr_text = ocr_image(client, model_name, str(file_path), config)

        # 保存结果
        combined_ocr_path = file_output_dir / "ocr_result.md"
        with open(combined_ocr_path, "w", encoding="utf-8") as f:
            f.write(f"# OCR 完整结果 - {file_path.name}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
            f.write(ocr_text)
            f.write("\n")

        elapsed = time.time() - start_time

        return {
            "file": file_path.name,
            "status": "success",
            "pages": 1,
            "elapsed_seconds": round(elapsed, 1),
            "output_dir": str(file_output_dir),
            "ocr_text": ocr_text,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] {file_path.name} 处理失败: {e}")
        import traceback
        traceback.print_exc()

        error_file = file_output_dir / "error.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"文件: {file_path.name}\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write(f"错误: {str(e)}\n")
            f.write(f"详情:\n{traceback.format_exc()}\n")

        return {
            "file": file_path.name,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 1),
        }





# =============================================================================
# 全局合并与统一结构化抽取
# =============================================================================

def merge_results(client: OpenAI, model_name: str, config: dict, output_dir: Path, results: list, output_formats: list):
    """合并所有 OCR 结果，并在长文本上进行统一全局抽取"""
    # 修复并发导致的乱序问题：合并前根据文件名重新排序
    results = sorted(results, key=lambda x: x["file"])
    
    # 废弃硬编码的 _merged 目录，直接在组目录的最外层输出
    merged_dir = output_dir
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    combined_ocr_for_extraction = []

    # 1. 合并所有 Markdown 文本
    if "markdown" in output_formats:
        merged_md = merged_dir / "merged_ocr.md"
        with open(merged_md, "w", encoding="utf-8") as f:
            f.write(f"# Qwen3.5-4B 批量 OCR 汇总\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"共处理 {len(results)} 个文件\n\n")
            f.write("---\n\n")

            for result in results:
                f.write(f"## {result['file']}\n\n")
                if result["status"] == "error":
                    f.write(f"**处理过程遇到错误**: {result.get('error', '未知错误')}\n\n")
                    f.write(f"*尝试强制合并已生成的 OCR 结果...*\n\n")

                file_dir = Path(result.get("output_dir", ""))
                if file_dir.exists():
                    md_files = sorted(file_dir.glob("ocr_result.md"))
                    for md_file in md_files:
                        try:
                            content = md_file.read_text(encoding="utf-8")
                            combined_ocr_for_extraction.append(f"【来源文件: {result['file']}】\n{content}")
                            f.write(content)
                            f.write("\n\n")
                        except Exception as e:
                            f.write(f"(读取失败: {e})\n\n")

                f.write("---\n\n")

        print(f"[MERGE] 所有文本已汇总至: {merged_md}")

        # 额外生成一个结构更干净的 txt 版本
        merged_txt = merged_dir / "merged_ocr.txt"
        with open(merged_txt, "w", encoding="utf-8") as f_txt:
            f_txt.write(f"合并时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in combined_ocr_for_extraction:
                f_txt.write(item)
                f_txt.write("\n\n========================================================================\n\n")
        print(f"[MERGE] 纯文本 txt 也已保存至: {merged_txt}")

    # 2. 对合并后的文本进行统一全局抽取
    if "json" in output_formats:
        merged_json = merged_dir / "merged_structured.json"
        full_text = "\n\n---\n\n".join(combined_ocr_for_extraction)
        
        extraction_config = config.get("extraction", {})
        global_structured = None
        
        if extraction_config.get("enabled", True) and full_text.strip():
            grp_name = output_dir.name
            print(f"\n[后台任务] 开始对 {grp_name} 进行全局 JSON 抽取 (文本长度: {len(full_text)} 字)...")
            start_ext = time.time()
            try:
                global_structured = extract_structured(client, model_name, full_text, config)
                ext_time = time.time() - start_ext
                print(f"\n[后台任务] ✅ {grp_name} 的 JSON 抽取成功！(耗时: {ext_time:.1f} 秒)")
            except Exception as e:
                ext_time = time.time() - start_ext
                print(f"\n[后台任务] ❌ {grp_name} 的 JSON 抽取失败: {e} (耗时: {ext_time:.1f} 秒)")
                global_structured = {"_error": str(e), "_text_length": len(full_text)}

        # 直接将干净的结构化 JSON 写入文件，不添加多余包裹信息
        with open(merged_json, "w", encoding="utf-8") as f:
            if global_structured is not None:
                import json
                json.dump(global_structured, f, ensure_ascii=False, indent=2)
            else:
                import json
                json.dump({"_warning": "No data extracted"}, f, ensure_ascii=False, indent=2)

        print(f"[MERGE] 全局抽取 JSON 结果已保存至: {merged_json}")



    print(f"[MERGE] 批处理摘要: {summary_path}")


# =============================================================================
# 归档已处理文件
# =============================================================================

def archive_file(file_path: Path, processed_dir: Path):
    """将已处理文件移到 processed 目录"""
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / file_path.name

    if dest.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = processed_dir / f"{stem}_{timestamp}{suffix}"

    shutil.move(str(file_path), str(dest))
    print(f"[ARCHIVE] {file_path.name} -> processed/{dest.name}")


# =============================================================================
# VLM Server 连接检测
# =============================================================================

def wait_for_vlm_server(server_url: str, max_retries: int = 30, interval: int = 10):
    """等待 vLLM Server 就绪"""
    import requests

    models_url = server_url.rstrip("/")
    if not models_url.endswith("/models"):
        models_url += "/models"

    print(f"[INFO] 等待 vLLM Server 就绪: {models_url}")

    for i in range(max_retries):
        try:
            resp = requests.get(models_url, timeout=5)
            if resp.status_code == 200:
                print(f"[INFO] vLLM Server 已就绪!")
                return True
        except Exception:
            pass

        print(f"[INFO] vLLM Server 尚未就绪, 等待中... ({i+1}/{max_retries})")
        time.sleep(interval)

    print(f"[ERROR] vLLM Server 未在 {max_retries * interval}s 内就绪")
    return False


# =============================================================================
# 图像预处理 (OpenCV)
# =============================================================================

def preprocess_single_image(file_path: Path, temp_dir: Path) -> Path:
    """流水线化的 OpenCV 预处理。每个 CPU 线程只处理自己分发到的单张图片。"""
    import cv2
    import numpy as np
    
    if file_path.suffix.lower() == ".pdf":
        return file_path
        
    img = cv2.imread(str(file_path))
    if img is None:
        return file_path
        
    debug_dir = temp_dir / file_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    current_img = img.copy()
    cv2.imwrite(str(debug_dir / "step0_original.jpg"), current_img)

    gaussian = cv2.GaussianBlur(current_img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(current_img, 1.5, gaussian, -0.5, 0)
    cv2.imwrite(str(debug_dir / "step1_sharpened.jpg"), sharpened)
    
    current_img = sharpened

    gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(debug_dir / "step2_gray.jpg"), gray)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(str(debug_dir / "step3_thresh.jpg"), thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(max_contour)
        img_area = current_img.shape[0] * current_img.shape[1]
        
        if max_area > img_area * 0.1:
            x, y, w, h = cv2.boundingRect(max_contour)
            margin_x = 5
            x_min = max(0, x - margin_x)
            x_max = min(gray.shape[1], x + w + margin_x)
            final_img = gray[:, x_min:x_max]
        else:
            final_img = gray
    else:
        final_img = gray
        
    cv2.imwrite(str(debug_dir / "step5_final_cropped.jpg"), final_img)
    
    final_ocr_input = temp_dir / file_path.name
    cv2.imwrite(str(final_ocr_input), final_img)
    
    return final_ocr_input


# =============================================================================
# 主入口
# =============================================================================

def main():
    print("=" * 70)
    print("  Qwen-VL 批处理系统 — 纯 OCR (全局并发版)")
    print("=" * 70)

    # 加载配置
    config = load_config()
    processing = config.get("processing", {})
    output_formats = processing.get("output_formats", ["markdown", "json"])
    merge_output = processing.get("merge_output", True)
    archive_processed = processing.get("archive_processed", True)
    supported_ext = processing.get("supported_extensions", [".jpg", ".jpeg", ".png", ".pdf"])

    # 路径配置
    input_dir = "/workspace/input"
    output_dir = Path("/workspace/output")
    processed_dir = Path(input_dir) / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # vLLM Server URL
    server_url = os.environ.get("VLLM_SERVER_URL", "http://vllm-server:8000/v1")

    # 等待 vLLM Server
    if not wait_for_vlm_server(server_url):
        print("[FATAL] 无法连接 vLLM Server, 退出")
        sys.exit(1)

    ocr_backend = "qwen-vl"
    print(f"\n[INFO] 使用 {ocr_backend} 作为 OCR 后端")

    # 同时初始化 OpenAI 客户端
    client = create_client(server_url)
    model_name = get_model_name(client)

    # 扫描文件并按目录分组
    grouped_files = scan_input_dir(input_dir, supported_ext)
    if not grouped_files:
        print("[INFO] 没有找到待处理的文件")
        sys.exit(0)

    total_files = sum(len(files) for files in grouped_files.values())
    print(f"\n[INFO] 找到 {len(grouped_files)} 个分组，共 {total_files} 个待处理文件")

    import concurrent.futures
    import threading

    # 自动满载调度系统
    extraction_workers = 3
    server_max_seqs = int(os.environ.get("MAX_NUM_SEQS", 16))
    ocr_workers = max(1, server_max_seqs - extraction_workers)
    
    print(f"\n[系统] 动态满载调度开启: 服务器总容量={server_max_seqs}")
    print(f"  -> 阶段一 (混合): OCR 并发={ocr_workers}, JSON 并发限制={extraction_workers}")
    print(f"  -> 阶段二 (纯享): OCR 结束后, JSON 并发将自动扩容至={server_max_seqs}")
    
    # JSON 并发节流阀 (在 OCR 运行期间，严格限制 JSON 的并发数)
    extraction_semaphore = threading.Semaphore(extraction_workers)
    
    # 1. 初始化全局线程池 (JSON 线程池的物理线程数直接开满，但受到 semaphore 逻辑限制)
    ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=ocr_workers)
    extraction_executor = concurrent.futures.ThreadPoolExecutor(max_workers=server_max_seqs)

    # 带有节流阀的后台提取任务包装器
    def extraction_task_with_semaphore(*args):
        with extraction_semaphore:
            with active_tasks["lock"]:
                active_tasks["json"] += 1
            try:
                merge_results(*args)
            finally:
                with active_tasks["lock"]:
                    active_tasks["json"] -= 1

    # 2. 初始化全局进度追踪器
    group_tracker = {}
    
    # 全局统计与实时在途任务追踪
    global_stats = {"success": 0, "error": 0, "lock": threading.Lock()}
    active_tasks = {"ocr": 0, "json": 0, "lock": threading.Lock()}
    
    # 3. 准备所有的任务
    print(f"\n{'-'*60}")
    print(f"[INFO] 启动全局流水线并发处理 (包含 OpenCV 预处理 + OCR)")
    print(f"{'-'*60}")

    for group_name, files in grouped_files.items():
        group_output_dir = output_dir / group_name
        group_processed_dir = processed_dir / group_name
        temp_dir = Path(input_dir) / ".tmp_cv2" / group_name
        
        # 提前清理临时文件夹
        import shutil
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        group_tracker[group_name] = {
            "total": len(files),
            "completed": 0,
            "results": [],
            "lock": threading.Lock(),
            "group_output_dir": group_output_dir,
            "group_processed_dir": group_processed_dir,
            "temp_dir": temp_dir
        }
        
    def _process_wrapper(orig_file, temp_dir, group_output_dir):
        with active_tasks["lock"]:
            active_tasks["ocr"] += 1
        try:
            # 流水线阶段 1: CPU 本地计算 OpenCV 图像预处理
            cv2_file = preprocess_single_image(orig_file, temp_dir)
            
            # 流水线阶段 2: 组装网络请求发送到 GPU vLLM
            res = process_single_file_qwen(
                client, model_name, cv2_file, group_output_dir, config
            )
            res["file"] = orig_file.name
            return orig_file, res
        finally:
            with active_tasks["lock"]:
                active_tasks["ocr"] -= 1

    all_ocr_futures = []

    # 无差别倾倒所有任务到全局线程池
    for group_name, files in grouped_files.items():
        tracker = group_tracker[group_name]
        for orig_file in sorted(files):
            future = ocr_executor.submit(_process_wrapper, orig_file, tracker["temp_dir"], tracker["group_output_dir"])
            
            # 使用闭包绑定当前组名
            def make_callback(g_name):
                def callback(f):
                    try:
                        orig_f, res = f.result()
                    except Exception as exc:
                        print(f"[ERROR] 发生致命并发异常: {exc}")
                        return
                        
                    t = group_tracker[g_name]
                    with t["lock"]:
                        t["results"].append(res)
                        t["completed"] += 1
                        
                        # 局部打印
                        if res["status"] == "success":
                            with global_stats["lock"]:
                                global_stats["success"] += 1
                            with active_tasks["lock"]:
                                cur_ocr = active_tasks["ocr"]
                                cur_json = active_tasks["json"]
                            print(f"[OCR] [{t['completed']}/{t['total']}] {g_name} - {orig_f.name} 完成 ({res['elapsed_seconds']}s) [当前 GPU 满载实况: {cur_ocr} OCR, {cur_json} JSON]")
                            if archive_processed:
                                archive_file(orig_f, t["group_processed_dir"])
                        else:
                            with global_stats["lock"]:
                                global_stats["error"] += 1
                            print(f"[OCR] ❌ {g_name} - {orig_f.name} 失败")

                        # 检查当前组是否全部完成
                        if t["completed"] == t["total"]:
                            print(f"\n[INFO] {g_name} 所有图片 OCR 处理完毕。")
                            if merge_output and t["results"]:
                                results = t["results"]
                                results.sort(key=lambda x: x.get("file", ""))
                                # 丢入后台 JSON 抽取队列 (受节流阀控制)
                                extraction_executor.submit(
                                    extraction_task_with_semaphore, client, model_name, config, t["group_output_dir"], results, output_formats
                                )
                return callback
                
            future.add_done_callback(make_callback(group_name))
            all_ocr_futures.append(future)

    # 4. 阻塞等待所有任务收尾
    # 等待所有的 OCR 任务执行完毕
    ocr_executor.shutdown(wait=True)
    
    print(f"\n{'='*70}")
    print(f"[INFO] 所有 OCR 流水线已收工！释放算力节流阀，JSON 长文本抽取火力全开 (并发提升至 {server_max_seqs})...")
    print(f"{'='*70}")
    
    # 释放所有被保留给 OCR 的并发额度，让积压的 JSON 任务抢占全部显卡算力
    for _ in range(server_max_seqs - extraction_workers):
        extraction_semaphore.release()
        
    # 等待所有的 JSON 任务执行完毕
    extraction_executor.shutdown(wait=True)

    # 最终报告
    print(f"\n{'='*70}")
    print(f"  全局批处理彻底完成!")
    total_success = global_stats["success"]
    total_error = global_stats["error"]
    print(f"  成功: {total_success} / {total_success + total_error}")
    print(f"  失败: {total_error} / {total_success + total_error}")
    print(f"  输出: {output_dir}")
    print(f"{'='*70}")

    if total_error > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
