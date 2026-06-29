# =============================================================================
# PP-DocLayoutV2 版面分析模块
# 使用 PaddlePaddle 的 PP-DocLayoutV2 对复杂排版（双列、表格等）进行版面检测，
# 按阅读顺序裁切子图，供 Qwen-VL 进行更精准的 OCR 识别。
# =============================================================================

import cv2
import numpy as np
from pathlib import Path


def _get_layout_model():
    """延迟加载 PP-DocLayoutV2 模型（全局单例）。"""
    global _layout_model_instance
    if "_layout_model_instance" not in globals() or _layout_model_instance is None:
        try:
            from paddleocr import LayoutDetection
            _layout_model_instance = LayoutDetection(model_name="PP-DocLayoutV2")
            print("[Layout] PP-DocLayoutV2 模型加载成功")
        except ImportError:
            raise RuntimeError(
                "PP-DocLayoutV2 需要安装 paddlepaddle 和 paddleocr。\n"
                "请运行: pip install paddlepaddle paddleocr[doc-parser]"
            )
    return _layout_model_instance

_layout_model_instance = None


def analyze_layout(image_path: Path, min_score: float = 0.5,
                   target_labels: list | None = None) -> list[dict]:
    """
    对单张图片做版面分析。

    Args:
        image_path: 图片路径
        min_score: 最低置信度阈值，低于该值的区域会被过滤
        target_labels: 只保留这些类型的区域 (如 ["text", "title", "table"])。
                       None 表示保留所有类型。

    Returns:
        按阅读顺序排列的区域列表：
        [
            {"label": "title", "bbox": [x1, y1, x2, y2], "score": 0.98, "order": 0},
            {"label": "text",  "bbox": [x1, y1, x2, y2], "score": 0.95, "order": 1},
            ...
        ]
    """
    model = _get_layout_model()
    output = model.predict(str(image_path), batch_size=1, layout_nms=True)

    regions = []
    for res in output:
        # res 内部按阅读顺序排列（Pointer Network 预测）
        if hasattr(res, "boxes") and res.boxes is not None:
            for idx, (box, label, score) in enumerate(
                zip(res.boxes, res.labels, res.scores)
            ):
                if score < min_score:
                    continue
                if target_labels and label.lower() not in [t.lower() for t in target_labels]:
                    continue
                regions.append({
                    "label": label,
                    "bbox": [int(b) for b in box],  # [x1, y1, x2, y2]
                    "score": float(score),
                    "order": idx,
                })
        # 兼容 dict 格式的输出
        elif hasattr(res, "__getitem__"):
            try:
                boxes = res.get("boxes", []) if isinstance(res, dict) else []
                for idx, item in enumerate(boxes):
                    score = item.get("score", 0)
                    label = item.get("label", "unknown")
                    bbox = item.get("coordinate", [0, 0, 0, 0])
                    if score < min_score:
                        continue
                    if target_labels and label.lower() not in [t.lower() for t in target_labels]:
                        continue
                    regions.append({
                        "label": label,
                        "bbox": [int(b) for b in bbox],
                        "score": float(score),
                        "order": idx,
                    })
            except Exception:
                pass

    # 确保按阅读顺序排序
    regions.sort(key=lambda r: r["order"])
    return regions


def crop_regions(image_path: Path, regions: list[dict],
                 output_dir: Path, padding: int = 5) -> list[Path]:
    """
    根据版面分析结果裁切子图，按阅读顺序返回路径列表。

    Args:
        image_path: 原始图片路径
        regions: analyze_layout 返回的区域列表
        output_dir: 裁切子图的输出目录
        padding: 裁切时在边界外扩展的像素数（防止文字被切断）

    Returns:
        按阅读顺序排列的子图路径列表
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[Layout] 无法读取图像: {image_path}")
        return [image_path]

    h, w = img.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_paths = []

    for i, region in enumerate(regions):
        x1, y1, x2, y2 = region["bbox"]
        # 添加 padding 并裁剪到图像边界
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # 跳过面积太小的区域（噪声）
        area = (x2 - x1) * (y2 - y1)
        if area < 100:
            continue

        cropped = img[y1:y2, x1:x2]
        out_path = output_dir / f"{image_path.stem}_r{i:02d}_{region['label']}.jpg"
        cv2.imwrite(str(out_path), cropped)
        cropped_paths.append(out_path)

    if not cropped_paths:
        # 如果没有检测到任何有效区域，返回原图
        print(f"[Layout] 未检测到有效区域，使用原图: {image_path.name}")
        return [image_path]

    print(f"[Layout] {image_path.name} -> 检测到 {len(cropped_paths)} 个区域")
    return cropped_paths
