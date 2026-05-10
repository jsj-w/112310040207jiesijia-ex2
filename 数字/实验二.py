from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "实验一输出" / "mnist_cnn_final.pth"


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
        raise SystemExit(
            "未安装 gradio。请先在 PyCharm 终端执行: pip install gradio pillow"
        ) from exc

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise SystemExit(
            "未安装 Pillow。请先在 PyCharm 终端执行: pip install pillow"
        ) from exc

    return gr, Image, ImageOps


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


def preprocess_image(image: Any) -> tuple[torch.Tensor, Any]:
    _, Image, ImageOps = ensure_web_dependencies()

    if image is None:
        raise ValueError("请先上传一张手写数字图片。")

    image = image.convert("L")
    image = ImageOps.exif_transpose(image)
    array = np.asarray(image, dtype=np.float32)

    # 大多数手机拍照或白底手写图像需要反色，转成 MNIST 的黑底白字风格。
    if array.mean() > 127:
        array = 255.0 - array

    array = np.clip(array, 0.0, 255.0) / 255.0
    mask = array > 0.12
    if mask.any():
        ys, xs = np.where(mask)
        array = array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    digit = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    width, height = digit.size
    scale = 20.0 / max(width, height, 1)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    digit = digit.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    canvas.paste(digit, offset)

    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, canvas


MODEL = None


def predict(image: Any) -> tuple[dict[str, float], str, Any]:
    global MODEL
    _, _, _ = ensure_web_dependencies()
    if MODEL is None:
        MODEL = load_model()

    tensor, preview = preprocess_image(image)
    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_digit = int(torch.argmax(probabilities).item())
    label_scores = {str(i): float(probabilities[i].item()) for i in range(10)}
    result_text = f"预测结果：{predicted_digit}"
    return label_scores, result_text, preview


def main() -> None:
    gr, _, _ = ensure_web_dependencies()

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="上传手写数字图片"),
        outputs=[
            gr.Label(num_top_classes=3, label="Top-3 预测概率"),
            gr.Textbox(label="预测类别"),
            gr.Image(type="pil", label="模型实际输入（28x28）"),
        ],
        title="实验二：MNIST 手写数字识别",
        description=(
            "先运行 实验一.py 生成模型文件，再在这里上传一张数字图片进行预测。"
            "建议使用白底黑字、数字尽量居中的图片。"
        ),
        allow_flagging="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
