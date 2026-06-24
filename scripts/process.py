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

    # 构建 Schema Template (NuExtract 风格)
    schema_lines = []
    for item in config.get("extraction", {}).get("schema", []):
        field = item.get("field", "")
        desc = item.get("description", "")
        # 如果描述里暗示了多键对象（字典）
        if "独立的键" in desc or "多个具体键" in desc or "作为键" in desc:
            template_block = (
                f'  "{field}": {{\n'
                f'    "__instruction__": "{desc}",\n'
                f'    "<填入动态识别的具体名称，例如：高血压>": {{\n'
                f'      "value": "<string, 提取出的具体数值或状态>",\n'
                f'      "evidence_id": <integer, 原文分句的数字编号>\n'
                f'    }}\n'
                f'  }}'
            )
        else:
            template_block = (
                f'  "{field}": {{\n'
                f'    "__instruction__": "{desc}",\n'
                f'    "value": "<string, 提取出的内容>",\n'
                f'    "evidence_id": <integer, 原文分句的数字编号>\n'
                f'  }}'
            )
        schema_lines.append(template_block)
    schema_template = "{\n" + ",\n".join(schema_lines) + "\n}"

    # 分割句子，保留标点符号，采用逗号级超细粒度切分以提升 Grounding 准度
    import re
    parts = re.split(r'([。！？\n，,；;])', ocr_text)
    sentences = ["".join(i).strip() for i in zip(parts[0::2], parts[1::2] + [""])]
    sentences = [s for s in sentences if s]
    
    # 构建带编号的文本
    numbered_lines = [f"[{i}] {s}" for i, s in enumerate(sentences)]
    numbered_text = "\n".join(numbered_lines)

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

    print("\n[JSON Extraction] 开始流式输出模型思考与提取过程：")
    print("-" * 50)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=inference.get("temperature", 0.0),
        max_tokens=inference.get("max_tokens", 4096),
        top_p=inference.get("top_p", 0.95),
        presence_penalty=inference.get("presence_penalty", 0.0),
        extra_body=extra_body,
        stream=True,
    )

    raw_response = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            raw_response += content
    print("\n" + "-" * 50)
    print("[JSON Extraction] 输出完毕。\n")

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

        # [Grounding] 后处理溯源：根据 evidence_id 还原完整原文
        def apply_grounding(data):
            if isinstance(data, dict):
                # 如果包含 evidence_id，执行精确替换
                if "evidence_id" in data:
                    try:
                        eid = int(data["evidence_id"])
                        if 0 <= eid < len(sentences):
                            data["evidence"] = sentences[eid]
                        else:
                            data["evidence"] = f"【未定位 ID:{eid}】"
                    except (ValueError, TypeError):
                        data["evidence"] = "【ID格式错误或无证据】"
                    
                    data.pop("evidence_id", None)
                    return
                
                # 递归遍历所有嵌套对象
                for k, v in data.items():
                    apply_grounding(v)
            elif isinstance(data, list):
                for item in data:
                    apply_grounding(item)
                    
        apply_grounding(parsed_json)
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
# OpenCV 图像预处理管线
# =============================================================================

def preprocess_images_with_opencv(file_paths: list, temp_dir: Path) -> list:
    """
    对图片进行 OpenCV 预处理：
    1. 灰度化
    2. 高斯模糊降噪
    3. 锐化
    4. 寻找最大轮廓并裁切（去除边框/黑边）
    5. 输出灰度裁切图像送入 OCR
    返回预处理后的图像路径列表
    """
    import cv2
    import numpy as np

    processed_files = []

    for file_path in file_paths:
        file_path = Path(file_path)
        
        # 为每个文件创建调试目录
        debug_dir = temp_dir / file_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 读取原始图像
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"    [WARN] 无法读取图像: {file_path.name}, 使用原始文件")
            processed_files.append(file_path)
            continue

        # 保存原图到调试目录
        cv2.imwrite(str(debug_dir / "0_original.jpg"), img)

        # ----------------
        # Step 1: 灰度化
        # ----------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(debug_dir / "1_gray.jpg"), gray)

        # ----------------
        # Step 2: 高斯模糊降噪
        # ----------------
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imwrite(str(debug_dir / "2_blurred.jpg"), blurred)

        # ----------------
        # Step 3: 锐化
        # ----------------
        sharpening_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, sharpening_kernel)
        cv2.imwrite(str(debug_dir / "3_sharpened.jpg"), sharpened)

        # ----------------
        # Step 4: 自适应二值化用于轮廓检测
        # ----------------
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        cv2.imwrite(str(debug_dir / "4_thresh.jpg"), thresh)

        # ----------------
        # Step 5: 寻找最大轮廓并裁切
        # ----------------
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_img = sharpened  # 使用锐化后的灰度图
        
        if contours:
            # 找到面积最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 如果轮廓面积占原图面积的比例大于 10%，才进行裁切
            img_area = img.shape[0] * img.shape[1]
            contour_area = w * h
            
            if contour_area / img_area > 0.10:
                # 仅裁切左右边缘，保留上下边缘 (根据要求取消了上下裁切)
                margin_x = 5
                x_min = max(0, x - margin_x)
                x_max = min(gray.shape[1], x + w + margin_x)
                
                # 使用灰度图进行裁切 (根据用户要求)
                final_img = gray[:, x_min:x_max]
            else:
                final_img = gray  # 轮廓太小, 使用完整灰度图
        else:
            final_img = gray  # 没找到轮廓, 使用完整灰度图

        cv2.imwrite(str(debug_dir / "5_final.jpg"), final_img)

        # 为了防止后续 OCR 输出目录冲突，将最终送去 OCR 的图片以原文件名保存在 temp_dir 根目录下
        final_ocr_input = temp_dir / file_path.name
        cv2.imwrite(str(final_ocr_input), final_img)
        
        # 打印调试信息
        print(f"    [OpenCV] {file_path.name} 流水线处理完毕，调试图保存于: {debug_dir}")
        
        processed_files.append(final_ocr_input)
        
    return processed_files



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

    print(f"[PROCESS] 开始处理: {file_path.name} (Qwen)")
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
        print(f"[DONE] {file_path.name} Qwen-VL 处理完成 ({elapsed:.1f}s)")

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
# 单文件处理 (PaddleOCR-VL Pipeline)
# =============================================================================

def process_single_file_paddleocr(
    pipeline,
    file_path: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """使用 PaddleOCR-VL Pipeline 处理单个文件（自动版面分析 + OCR）"""
    file_name = file_path.stem
    file_output_dir = output_dir / file_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PROCESS] 开始处理: {file_path.name}")
    start_time = time.time()

    try:
        # 提取推理配置
        inference_cfg = config.get("inference", {})
        repetition_penalty = float(inference_cfg.get("repetition_penalty", 1.1))
        temperature = float(inference_cfg.get("temperature", 0.0))
        top_p = float(inference_cfg.get("top_p", 1.0))
        
        # PaddleOCR-VL Pipeline 自动处理：版面检测 → 子区域裁切 → VLM OCR → Markdown 拼接
        # 并传入重复惩罚等采样参数，防止复读机现象
        output = pipeline.predict(
            str(file_path),
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p
        )
        
        all_ocr_texts = []
        page_count = 0
        
        for res in output:
            page_count += 1
            # 从 PaddleOCR 结果中提取 Markdown 文本
            # PaddleOCR-VL 的 predict 结果对象包含 text 属性
            if hasattr(res, 'text'):
                page_text = res.text
            elif hasattr(res, 'rec_text'):
                page_text = res.rec_text
            else:
                # 尝试通过 save_to_markdown 获取文本
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    res.save_to_markdown(save_path=tmpdir)
                    # 读取生成的 markdown 文件
                    md_files = list(Path(tmpdir).rglob("*.md"))
                    if md_files:
                        page_text = md_files[0].read_text(encoding="utf-8")
                    else:
                        page_text = str(res)
            
            all_ocr_texts.append(page_text)
            
            # 也让 PaddleOCR 保存原始结果到单文件输出目录
            try:
                res.save_to_json(save_path=str(file_output_dir))
                res.save_to_markdown(save_path=str(file_output_dir))
            except Exception as save_err:
                print(f"  [WARN] 保存 PaddleOCR 原始结果失败: {save_err}")

        # 保存合并的单文件结果 (ocr_result.md)
        combined_ocr = "\n\n---\n\n".join(all_ocr_texts)
        combined_ocr_path = file_output_dir / "ocr_result.md"
        with open(combined_ocr_path, "w", encoding="utf-8") as f:
            f.write(f"# OCR 完整结果 - {file_path.name}\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总页数: {page_count}\n\n---\n\n")
            f.write(combined_ocr)
            f.write("\n")

        elapsed = time.time() - start_time
        print(f"[DONE] {file_path.name} PaddleOCR-VL 处理完成 ({elapsed:.1f}s)")

        return {
            "file": file_path.name,
            "status": "success",
            "pages": page_count,
            "elapsed_seconds": round(elapsed, 1),
            "output_dir": str(file_output_dir),
            "ocr_text": combined_ocr,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] {file_path.name} 处理失败: {e}")
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
            print(f"[MERGE] 进行全局结构化抽取 (总文字长度: {len(full_text)} 字符)...")
            try:
                global_structured = extract_structured(client, model_name, full_text, config)
            except Exception as e:
                print(f"[ERROR] 全局结构化抽取失败: {e}")
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

    # 3. 摘要报告
    summary_path = merged_dir / "summary.json"
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_files": len(results),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
        "files": [
            {
                "file": r["file"],
                "status": r["status"],
                "pages": r.get("pages", 0),
                "elapsed_seconds": r.get("elapsed_seconds", 0),
            }
            for r in results
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

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

def preprocess_images_with_opencv(files: list, temp_dir: Path) -> list:
    """使用 OpenCV 处理图片流水线：增强对比度 + 去除非文字边缘"""
    import cv2
    import numpy as np
    import shutil

    if temp_dir.exists():
        shutil.rmtree(str(temp_dir))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    
    for file_path in files:
        if file_path.suffix.lower() == ".pdf":
            # PDF暂不走CV处理
            processed_files.append(file_path)
            continue
            
        img = cv2.imread(str(file_path))
        if img is None:
            processed_files.append(file_path)
            continue
            
        # 为当前图片创建专属调试目录
        debug_dir = temp_dir / file_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # -------------------------------------------------------------
        # 基础画面预处理流水线 (细分步骤供 Debug)
        # -------------------------------------------------------------
        current_img = img.copy()
        cv2.imwrite(str(debug_dir / "step0_original.jpg"), current_img)

        # Step 1: 仅图像锐化 (Unsharp Masking)
        # 提升文字清晰度，对抗相机的轻微失焦，移除可能引起失真的对比度增强
        gaussian = cv2.GaussianBlur(current_img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(current_img, 1.5, gaussian, -0.5, 0)
        cv2.imwrite(str(debug_dir / "step1_sharpened.jpg"), sharpened)
        
        # 将锐化后的图作为主图
        current_img = sharpened

        # -------------------------------------------------------------
        # Step 2: 提取灰度图
        # -------------------------------------------------------------
        gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(debug_dir / "step2_gray.jpg"), gray)
        
        # -------------------------------------------------------------
        # Step 3: 全局大津法二值化 (直接寻找文档亮区)
        # -------------------------------------------------------------
        # 根据需求：无高斯模糊，直接二值化，将明亮的屏幕页面与暗色的屏幕边框剥离
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(str(debug_dir / "step3_thresh.jpg"), thresh)
        
        # -------------------------------------------------------------
        # Step 4: 寻找最大亮区并裁剪边缘
        # -------------------------------------------------------------
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 找到面积最大的轮廓，即文档主体区域
            max_contour = max(contours, key=cv2.contourArea)
            max_area = cv2.contourArea(max_contour)
            img_area = current_img.shape[0] * current_img.shape[1]
            
            # 最大轮廓需大于一定比例，防止误切
            if max_area > img_area * 0.1:
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # 仅裁切左右边缘，保留上下边缘 (根据要求取消了上下裁切)
                margin_x = 5
                x_min = max(0, x - margin_x)
                x_max = min(gray.shape[1], x + w + margin_x)
                
                final_img = gray[:, x_min:x_max]
            else:
                final_img = gray
        else:
            final_img = gray
            
        cv2.imwrite(str(debug_dir / "step5_final_cropped.jpg"), final_img)
        
        # 为了防止后续 OCR 输出目录冲突，将最终送去 OCR 的图片以原文件名保存在 temp_dir 根目录下
        final_ocr_input = temp_dir / file_path.name
        cv2.imwrite(str(final_ocr_input), final_img)
        
        # 打印调试信息
        print(f"    [OpenCV] {file_path.name} 流水线处理完毕，调试图保存于: {debug_dir}")
        
        processed_files.append(final_ocr_input)
        
    return processed_files


# =============================================================================
# 主入口
# =============================================================================

def main():
    print("=" * 70)
    print("  PaddleOCR-VL-1.6 批处理系统 — PPLayout 版面分析 + OCR")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    ocr_backend = config.get("ocr", {}).get("backend", "qwen-vl")
    
    if ocr_backend == "paddleocr-vl":
        # 使用 ThreadLocal 管理 PaddleOCR-VL Pipeline，避免 C++ 并发底层报错 (double free)
        print(f"\n[INFO] 配置 PaddleOCR-VL-1.6 Pipeline (延迟到每个线程独立初始化)...")
        print(f"[INFO] VLM 推理后端: vLLM Server @ {server_url}")
        
        import threading
        thread_local = threading.local()

        def get_thread_local_pipeline():
            if not hasattr(thread_local, "pipeline"):
                from paddleocr import PaddleOCRVL
                thread_local.pipeline = PaddleOCRVL(
                    pipeline_version="v1.6",
                    vl_rec_backend="vllm-server",
                    vl_rec_server_url=server_url,
                )
                print(f"    [Thread-{threading.get_ident()}] Pipeline 初始化完成")
            return thread_local.pipeline

    else:
        print(f"\n[INFO] 使用 {ocr_backend} 作为 OCR 后端")
        thread_local = None
        
        def get_thread_local_pipeline():
            return None

    # 同时初始化 OpenAI 客户端（用于后续的结构化提取，如果需要的话）
    client = create_client(server_url)
    model_name = get_model_name(client)

    # 扫描文件并按目录分组
    grouped_files = scan_input_dir(input_dir, supported_ext)
    if not grouped_files:
        print("[INFO] 没有找到待处理的文件")
        sys.exit(0)

    total_files = sum(len(files) for files in grouped_files.values())
    print(f"\n[INFO] 找到 {len(grouped_files)} 个分组，共 {total_files} 个待处理文件")

    total_success = 0
    total_error = 0

    # 外层循环：遍历所有病人/分组
    for group_name, files in grouped_files.items():
        print(f"\n{'='*70}")
        print(f"  开始处理分组: {group_name} (包含 {len(files)} 个文件)")
        print(f"{'='*70}")
        
        # 为当前分组分配特定的输出目录
        group_output_dir = output_dir / group_name
        group_processed_dir = processed_dir / group_name
        temp_dir = Path(input_dir) / ".tmp_cv2" / group_name

        # Step 1: OpenCV 图像预处理（裁边、灰度化等）
        print(f"\n[INFO] 启动 OpenCV 图像预处理...")
        processed_files = preprocess_images_with_opencv(files, temp_dir)

        # Step 2: 并发 OCR 处理
        results = []
        success_count = 0
        error_count = 0

        import concurrent.futures

        max_workers = processing.get("max_workers", 4)
        print(f"\n{'-'*60}")
        print(f"[INFO] 启动并发 OCR 处理 (最大并发数: {max_workers}, 后端: {ocr_backend})")
        print(f"{'-'*60}")

        # 使用线程池并发处理当前分组的文件
        def _process_wrapper(args):
            idx, total, orig_file, cv2_file = args
            print(f"\n[{idx}/{total}] 开始处理: {orig_file.name}")
            
            if ocr_backend == "paddleocr-vl":
                # 获取当前线程专属的 pipeline 实例，避免 C++ 并发报错
                local_pipeline = get_thread_local_pipeline()
                res = process_single_file_paddleocr(
                    local_pipeline, cv2_file, group_output_dir, config
                )

            else:
                res = process_single_file_qwen(
                    client, model_name, cv2_file, group_output_dir, config
                )
                
            res["file"] = orig_file.name
            return orig_file, res

        tasks_args = [
            (idx, len(files), orig_file, cv2_file) 
            for idx, (orig_file, cv2_file) in enumerate(zip(sorted(files), processed_files), 1)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_process_wrapper, args): args[2] 
                for args in tasks_args
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                orig_file = future_to_file[future]
                try:
                    _, result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        success_count += 1
                        if archive_processed:
                            archive_file(orig_file, group_processed_dir)
                    else:
                        error_count += 1
                except Exception as exc:
                    print(f"[ERROR] {orig_file.name} 并发处理时抛出异常: {exc}")
                    error_count += 1

        total_success += success_count
        total_error += error_count

        # 合并结果与全局抽取 (阶段二: 结构化抽取)
        if merge_output and results:
            print(f"\n{'-'*60}")
            print(f"[MERGE] {group_name}: 汇总结果并执行独立结构化抽取...")
            
            # 由于并发执行返回顺序不确定，必须按文件名重新排序，保证病历页码连续性
            results.sort(key=lambda x: x.get("file", ""))
            
            merge_results(client, model_name, config, group_output_dir, results, output_formats)
            
        print(f"\n[INFO] {group_name} 分组处理完成，调试图像保存在 {temp_dir} 下。")

    # 最终报告
    print(f"\n{'='*70}")
    print(f"  批处理完成!")
    print(f"  成功: {total_success} / {total_success + total_error}")
    print(f"  失败: {total_error} / {total_success + total_error}")
    print(f"  输出: {output_dir}")
    print(f"{'='*70}")

    if total_error > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
