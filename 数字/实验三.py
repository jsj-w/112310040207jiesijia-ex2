from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "实验一输出" / "mnist_cnn_final.pth"
PREDICTION_HISTORY: list[list[str]] = []
MODEL: nn.Module | None = None


class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def ensure_web_dependencies() -> tuple[Any, Any, Any]:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("未安装 gradio，请先执行: pip install gradio pillow") from exc

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SystemExit("未安装 Pillow，请先执行: pip install pillow") from exc

    return gr, Image, ImageOps


def get_resample_filter(image_module: Any) -> Any:
    if hasattr(image_module, "Resampling"):
        return image_module.Resampling.LANCZOS
    return image_module.LANCZOS


def load_model(model_path: Path = MODEL_PATH) -> SimpleCNN:
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件: {model_path}\n请先运行 实验一.py 训练并生成 mnist_cnn_final.pth"
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    dropout = float(config.get("dropout", 0.0))
    model = SimpleCNN(dropout=dropout)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_canvas_input(canvas_data: Any) -> Any:
    _, Image, _ = ensure_web_dependencies()

    if canvas_data is None:
        return None

    if hasattr(canvas_data, "convert"):
        return canvas_data

    if isinstance(canvas_data, dict):
        for key in ("composite", "image", "background"):
            value = canvas_data.get(key)
            if value is not None:
                normalized = normalize_canvas_input(value)
                if normalized is not None:
                    return normalized
        layers = canvas_data.get("layers")
        if isinstance(layers, list):
            for layer in reversed(layers):
                normalized = normalize_canvas_input(layer)
                if normalized is not None:
                    return normalized
        return None

    if isinstance(canvas_data, np.ndarray):
        array = canvas_data.astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return Image.fromarray(array)
        raise ValueError("画板返回了不支持的数组形状。")

    raise ValueError(f"无法解析当前画板输出类型: {type(canvas_data).__name__}")


def pil_to_grayscale_array(image: Any) -> np.ndarray:
    _, Image, ImageOps = ensure_web_dependencies()
    image = ImageOps.exif_transpose(image)

    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("L")
    else:
        image = image.convert("L")

    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def preprocess_image(canvas_data: Any) -> tuple[torch.Tensor, Any]:
    _, Image, _ = ensure_web_dependencies()
    image = normalize_canvas_input(canvas_data)
    if image is None:
        raise ValueError("请先在画板中写一个数字。")

    array = pil_to_grayscale_array(image)

    # 同时兼容白底黑字和黑底白字。
    if array.mean() > 0.5:
        array = 1.0 - array

    array = np.clip(array, 0.0, 1.0)
    mask = array > 0.10
    if not mask.any():
        raise ValueError("没有检测到清晰的笔迹，请写得更粗一点或更居中一些。")

    ys, xs = np.where(mask)
    array = array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    digit = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    width, height = digit.size
    scale = 20.0 / max(width, height, 1)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    digit = digit.resize(new_size, get_resample_filter(Image))

    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    canvas.paste(digit, offset)

    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, canvas


def build_probability_html(probabilities: list[float]) -> str:
    bars = []
    for digit, prob in enumerate(probabilities):
        width = max(prob * 100.0, 1.5)
        bars.append(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
              <div style="width:18px;font-weight:700;">{digit}</div>
              <div style="flex:1;background:#e5e7eb;border-radius:999px;overflow:hidden;height:16px;">
                <div style="width:{width:.2f}%;height:16px;background:linear-gradient(90deg,#0ea5e9,#22c55e);"></div>
              </div>
              <div style="width:72px;text-align:right;">{prob * 100:.2f}%</div>
            </div>
            """
        )
    return (
        "<div style='padding:12px;border-radius:14px;background:#f8fafc;'>"
        "<div style='font-weight:700;margin-bottom:10px;'>各类别概率分布</div>"
        + "".join(bars)
        + "</div>"
    )


def empty_outputs(message: str = "请先在左侧画板中写一个数字。") -> tuple[str, dict[str, float], Any, str, list[list[str]]]:
    return (
        message,
        {str(i): 0.0 for i in range(10)},
        None,
        "<div style='padding:12px;border-radius:14px;background:#f8fafc;'>等待识别结果...</div>",
        PREDICTION_HISTORY[-10:],
    )


def predict_from_canvas(canvas_data: Any) -> tuple[str, dict[str, float], Any, str, list[list[str]]]:
    global MODEL

    try:
        if MODEL is None:
            MODEL = load_model()

        tensor, preview = preprocess_image(canvas_data)
        with torch.no_grad():
            logits = MODEL(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        probs = [float(probabilities[i].item()) for i in range(10)]
        predicted_digit = int(torch.argmax(probabilities).item())
        top3 = torch.topk(probabilities, k=3)
        top3_text = " | ".join(
            f"{int(idx)}: {float(score) * 100:.2f}%"
            for score, idx in zip(top3.values.tolist(), top3.indices.tolist())
        )
        result_text = f"识别结果：{predicted_digit}\nTop-3：{top3_text}"
        label_scores = {str(i): probs[i] for i in range(10)}
        probability_html = build_probability_html(probs)

        PREDICTION_HISTORY.append(
            [
                datetime.now().strftime("%H:%M:%S"),
                str(predicted_digit),
                f"{max(probs) * 100:.2f}%",
                top3_text,
            ]
        )
        history_rows = PREDICTION_HISTORY[-10:]
        return result_text, label_scores, preview, probability_html, history_rows
    except Exception as exc:
        return empty_outputs(f"识别失败：{exc}")


def clear_canvas_and_outputs() -> tuple[Any, str, dict[str, float], Any, str]:
    return (
        None,
        "画板已清空，请重新书写数字。",
        {str(i): 0.0 for i in range(10)},
        None,
        "<div style='padding:12px;border-radius:14px;background:#f8fafc;'>等待识别结果...</div>",
    )


def clear_history() -> list[list[str]]:
    PREDICTION_HISTORY.clear()
    return []


def build_canvas_component(gr: Any) -> Any:
    if hasattr(gr, "Sketchpad"):
        return gr.Sketchpad(label="手写画板", height=320)
    if hasattr(gr, "ImageEditor"):
        return gr.ImageEditor(label="手写画板", height=320)
    return gr.Image(type="pil", label="手写输入区域")


def main() -> None:
    gr, _, _ = ensure_web_dependencies()

    with gr.Blocks(title="实验三：交互式手写识别系统") as demo:
        gr.Markdown(
            """
            # 实验三：交互式手写识别系统
            在左侧画板中直接手写数字，点击“开始识别”即可得到预测结果。
            建议把数字写粗一点、尽量居中，识别会更稳定。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                canvas = build_canvas_component(gr)
                with gr.Row():
                    predict_button = gr.Button("开始识别", variant="primary")
                    clear_button = gr.Button("清空画板")
                    clear_history_button = gr.Button("清空历史")
            with gr.Column(scale=1):
                result_text = gr.Textbox(label="识别结果", lines=3, value="请先在左侧画板中写一个数字。")
                label_output = gr.Label(num_top_classes=3, label="Top-3 概率")
                processed_preview = gr.Image(type="pil", label="模型实际输入（28x28）")
                probability_html = gr.HTML("<div style='padding:12px;border-radius:14px;background:#f8fafc;'>等待识别结果...</div>")

        history_table = gr.Dataframe(
            headers=["时间", "预测数字", "最高置信度", "Top-3"],
            datatype=["str", "str", "str", "str"],
            row_count=10,
            col_count=(4, "fixed"),
            label="历史识别记录",
            value=[],
        )

        predict_button.click(
            fn=predict_from_canvas,
            inputs=canvas,
            outputs=[result_text, label_output, processed_preview, probability_html, history_table],
        )
        clear_button.click(
            fn=clear_canvas_and_outputs,
            outputs=[canvas, result_text, label_output, processed_preview, probability_html],
        )
        clear_history_button.click(fn=clear_history, outputs=history_table)

    demo.launch()


if __name__ == "__main__":
    main()
