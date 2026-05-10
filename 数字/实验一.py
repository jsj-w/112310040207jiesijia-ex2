from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "实验一输出"
TRAIN_CSV = BASE_DIR / "train.csv"
TEST_CSV = BASE_DIR / "test.csv"
TRAIN_CACHE = OUTPUT_DIR / "train_cache.pt"
TEST_CACHE = OUTPUT_DIR / "test_cache.pt"
RESULTS_CSV = OUTPUT_DIR / "对比实验结果.csv"
RESULTS_JSON = OUTPUT_DIR / "对比实验结果.json"
LOSS_HISTORY_JSON = OUTPUT_DIR / "loss历史.json"
LOSS_PLOT = OUTPUT_DIR / "loss曲线.png"
FINAL_MODEL_PATH = OUTPUT_DIR / "mnist_cnn_final.pth"
KAGGLE_SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
FINAL_SUMMARY_PATH = OUTPUT_DIR / "最终模型信息.json"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass
class ExperimentConfig:
    name: str
    optimizer: str
    learning_rate: float
    batch_size: int
    augment: bool
    early_stopping: bool
    epochs: int
    patience: int = 3
    scheduler: bool = False
    weight_decay: float = 0.0
    dropout: float = 0.0
    note: str = ""


class KaggleMNISTDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index: int) -> Any:
        image = self.images[index]
        if self.labels is None:
            return image
        return image, self.labels[index]


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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_train_data() -> tuple[torch.Tensor, torch.Tensor]:
    ensure_output_dir()
    if TRAIN_CACHE.exists():
        cached = torch.load(TRAIN_CACHE, map_location="cpu")
        return cached["images"], cached["labels"]

    print("首次读取 train.csv，正在生成缓存...")
    raw = np.loadtxt(TRAIN_CSV, delimiter=",", skiprows=1, dtype=np.uint8)
    labels = torch.from_numpy(raw[:, 0].astype(np.int64))
    images = torch.from_numpy(raw[:, 1:].astype(np.float32).reshape(-1, 1, 28, 28) / 255.0)
    torch.save({"images": images, "labels": labels}, TRAIN_CACHE)
    return images, labels


def load_test_data() -> torch.Tensor:
    ensure_output_dir()
    if TEST_CACHE.exists():
        cached = torch.load(TEST_CACHE, map_location="cpu")
        return cached["images"]

    print("首次读取 test.csv，正在生成缓存...")
    raw = np.loadtxt(TEST_CSV, delimiter=",", skiprows=1, dtype=np.uint8)
    images = torch.from_numpy(raw.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0)
    torch.save({"images": images}, TEST_CACHE)
    return images


def split_dataset(
    images: torch.Tensor,
    labels: torch.Tensor,
    max_train_samples: int | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    total = images.size(0)
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(SEED))

    if max_train_samples is not None:
        max_train_samples = min(max_train_samples, total)
        indices = indices[:max_train_samples]
        images = images[indices]
        labels = labels[indices]
        total = max_train_samples
        indices = torch.arange(total)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "train": (images[train_idx], labels[train_idx]),
        "val": (images[val_idx], labels[val_idx]),
        "test": (images[test_idx], labels[test_idx]),
    }


def build_loaders(
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> dict[str, DataLoader]:
    train_dataset = KaggleMNISTDataset(*splits["train"])
    val_dataset = KaggleMNISTDataset(*splits["val"])
    test_dataset = KaggleMNISTDataset(*splits["test"])

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "train_eval": DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def apply_batch_augmentation(
    images: torch.Tensor,
    max_degrees: float = 10.0,
    max_translate_ratio: float = 0.1,
) -> torch.Tensor:
    batch_size = images.size(0)
    device = images.device
    angles = (torch.rand(batch_size, device=device) * 2 - 1) * math.radians(max_degrees)
    max_shift = max_translate_ratio * 2.0
    shifts_x = (torch.rand(batch_size, device=device) * 2 - 1) * max_shift
    shifts_y = (torch.rand(batch_size, device=device) * 2 - 1) * max_shift

    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = torch.cos(angles)
    theta[:, 0, 1] = -torch.sin(angles)
    theta[:, 1, 0] = torch.sin(angles)
    theta[:, 1, 1] = torch.cos(angles)
    theta[:, 0, 2] = shifts_x
    theta[:, 1, 2] = shifts_y

    grid = F.affine_grid(theta, images.size(), align_corners=False)
    augmented = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return augmented.clamp_(0.0, 1.0)


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"不支持的优化器: {config.optimizer}")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_one_experiment(
    config: ExperimentConfig,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    loaders = build_loaders(splits, config.batch_size)
    model = SimpleCNN(dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = None
    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }

    print(
        f"\n开始训练 {config.name}: optimizer={config.optimizer}, "
        f"lr={config.learning_rate}, batch_size={config.batch_size}, "
        f"augment={config.augment}, early_stopping={config.early_stopping}"
    )

    started_at = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)

            if config.augment:
                images = apply_batch_augmentation(images)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        print(
            f"[{config.name}] Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        improved = val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.early_stopping and epochs_without_improvement >= config.patience:
            print(f"[{config.name}] 触发 Early Stopping，停止于 epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval_loss, train_eval_acc = evaluate(model, loaders["train_eval"], criterion, device)
    val_eval_loss, val_eval_acc = evaluate(model, loaders["val"], criterion, device)
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
    elapsed = time.time() - started_at

    result = {
        "name": config.name,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "augment": config.augment,
        "early_stopping": config.early_stopping,
        "scheduler": config.scheduler,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "train_acc": round(train_eval_acc, 4),
        "val_acc": round(val_eval_acc, 4),
        "test_acc": round(test_acc, 4),
        "lowest_loss": round(best_val_loss, 4),
        "converged_epoch": best_epoch,
        "train_loss": round(train_eval_loss, 4),
        "val_loss": round(val_eval_loss, 4),
        "test_loss": round(test_loss, 4),
        "elapsed_seconds": round(elapsed, 2),
        "history": history,
        "state_dict": model.state_dict(),
        "config": asdict(config),
    }
    return result


def save_comparison_results(results: list[dict[str, Any]]) -> None:
    rows = []
    for item in results:
        rows.append(
            {
                "实验编号": item["name"],
                "Train Acc": item["train_acc"],
                "Val Acc": item["val_acc"],
                "Test Acc": item["test_acc"],
                "最低 Loss": item["lowest_loss"],
                "收敛 Epoch": item["converged_epoch"],
                "优化器": item["optimizer"],
                "学习率": item["learning_rate"],
                "Batch Size": item["batch_size"],
                "数据增强": "是" if item["augment"] else "否",
                "Early Stopping": "是" if item["early_stopping"] else "否",
            }
        )

    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    serializable_results = []
    for item in results:
        cleaned = {k: v for k, v in item.items() if k not in {"state_dict"}}
        serializable_results.append(cleaned)

    RESULTS_JSON.write_text(
        json.dumps(serializable_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOSS_HISTORY_JSON.write_text(
        json.dumps({item["name"]: item["history"] for item in results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_loss_curves(results: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未检测到 matplotlib，已跳过 loss 曲线绘制。安装后重新运行即可生成图片。")
        return

    plt.figure(figsize=(10, 6))
    for item in results:
        history = item["history"]
        plt.plot(history["epoch"], history["train_loss"], marker="o", label=item["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST CNN Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=200)
    plt.close()
    print(f"已保存 loss 曲线图: {LOSS_PLOT}")


def choose_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=lambda item: (item["val_acc"], item["test_acc"], -item["lowest_loss"]))


def save_final_model(final_result: dict[str, Any]) -> None:
    checkpoint = {
        "model_state_dict": final_result["state_dict"],
        "config": final_result["config"],
        "metrics": {
            "train_acc": final_result["train_acc"],
            "val_acc": final_result["val_acc"],
            "test_acc": final_result["test_acc"],
            "lowest_loss": final_result["lowest_loss"],
            "converged_epoch": final_result["converged_epoch"],
        },
    }
    torch.save(checkpoint, FINAL_MODEL_PATH)

    summary = {
        "optimizer": final_result["optimizer"],
        "learning_rate": final_result["learning_rate"],
        "batch_size": final_result["batch_size"],
        "epochs": final_result["config"]["epochs"],
        "augment": final_result["augment"],
        "early_stopping": final_result["early_stopping"],
        "scheduler": final_result["scheduler"],
        "weight_decay": final_result["weight_decay"],
        "dropout": final_result["dropout"],
        "train_acc": final_result["train_acc"],
        "val_acc": final_result["val_acc"],
        "test_acc": final_result["test_acc"],
        "note": final_result["config"]["note"],
    }
    FINAL_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存最终模型: {FINAL_MODEL_PATH}")


def make_submission(
    model_state_dict: dict[str, torch.Tensor],
    dropout: float,
    device: torch.device,
    batch_size: int = 256,
) -> None:
    model = SimpleCNN(dropout=dropout).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    test_images = load_test_data()
    test_loader = DataLoader(KaggleMNISTDataset(test_images), batch_size=batch_size, shuffle=False)
    predictions: list[int] = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    with KAGGLE_SUBMISSION_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId", "Label"])
        for idx, label in enumerate(predictions, start=1):
            writer.writerow([idx, label])

    print(f"已生成 Kaggle 提交文件: {KAGGLE_SUBMISSION_PATH}")


def get_experiment_configs(compare_epochs: int, final_epochs: int) -> tuple[list[ExperimentConfig], ExperimentConfig]:
    comparison_configs = [
        ExperimentConfig(
            name="Exp1",
            optimizer="SGD",
            learning_rate=0.01,
            batch_size=64,
            augment=False,
            early_stopping=False,
            epochs=compare_epochs,
            note="基础对照组",
        ),
        ExperimentConfig(
            name="Exp2",
            optimizer="Adam",
            learning_rate=0.001,
            batch_size=64,
            augment=False,
            early_stopping=False,
            epochs=compare_epochs,
            note="仅替换为 Adam",
        ),
        ExperimentConfig(
            name="Exp3",
            optimizer="Adam",
            learning_rate=0.001,
            batch_size=128,
            augment=False,
            early_stopping=True,
            epochs=compare_epochs,
            patience=3,
            note="更大 batch 并启用 Early Stopping",
        ),
        ExperimentConfig(
            name="Exp4",
            optimizer="Adam",
            learning_rate=0.001,
            batch_size=64,
            augment=True,
            early_stopping=True,
            epochs=compare_epochs,
            patience=3,
            note="加入旋转和平移增强",
        ),
    ]

    final_config = ExperimentConfig(
        name="Final",
        optimizer="Adam",
        learning_rate=0.001,
        batch_size=128,
        augment=True,
        early_stopping=True,
        epochs=final_epochs,
        patience=5,
        scheduler=True,
        weight_decay=1e-4,
        dropout=0.25,
        note="推荐用于 Kaggle 提交的最终配置",
    )
    return comparison_configs, final_config


def print_result_table(results: list[dict[str, Any]]) -> None:
    print("\n对比实验结果汇总:")
    print("-" * 92)
    print(
        f"{'实验':<8}{'Train Acc':<12}{'Val Acc':<12}{'Test Acc':<12}"
        f"{'最低Loss':<12}{'收敛Epoch':<12}{'耗时(s)':<12}"
    )
    print("-" * 92)
    for item in results:
        print(
            f"{item['name']:<8}{item['train_acc']:<12.4f}{item['val_acc']:<12.4f}"
            f"{item['test_acc']:<12.4f}{item['lowest_loss']:<12.4f}"
            f"{item['converged_epoch']:<12}{item['elapsed_seconds']:<12.2f}"
        )
    print("-" * 92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST CNN 对比实验与提交文件生成")
    parser.add_argument(
        "--mode",
        choices=["all", "compare", "final"],
        default="all",
        help="all: 运行对比实验并训练最终模型；compare: 仅对比实验；final: 仅训练最终模型",
    )
    parser.add_argument("--compare-epochs", type=int, default=12, help="四组对比实验的训练轮数")
    parser.add_argument("--final-epochs", type=int, default=20, help="最终模型的训练轮数")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="调试用，只取前 N 个样本参与训练",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="训练设备选择",
    )
    return parser.parse_args()


def resolve_device(device_option: str) -> torch.device:
    if device_option == "cpu":
        return torch.device("cpu")
    if device_option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用 CUDA，请改为 --device cpu 或 --device auto")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    ensure_output_dir()
    set_seed(SEED)
    device = resolve_device(args.device)
    print(f"当前设备: {device}")

    images, labels = load_train_data()
    splits = split_dataset(images, labels, max_train_samples=args.max_train_samples)
    compare_configs, final_config = get_experiment_configs(args.compare_epochs, args.final_epochs)

    comparison_results: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None

    if args.mode in {"all", "compare"}:
        for config in compare_configs:
            comparison_results.append(train_one_experiment(config, splits, device))
        print_result_table(comparison_results)
        save_comparison_results(comparison_results)
        plot_loss_curves(comparison_results)

    if args.mode in {"all", "final"}:
        final_result = train_one_experiment(final_config, splits, device)
        save_final_model(final_result)
        make_submission(
            model_state_dict=final_result["state_dict"],
            dropout=final_result["dropout"],
            device=device,
            batch_size=final_result["batch_size"],
        )
        print(
            "\n最终模型结果: "
            f"train_acc={final_result['train_acc']:.4f}, "
            f"val_acc={final_result['val_acc']:.4f}, "
            f"test_acc={final_result['test_acc']:.4f}"
        )

    if comparison_results and final_result is None:
        best = choose_best_result(comparison_results)
        print(f"\n当前对比实验中表现最好的是 {best['name']}，验证集准确率 {best['val_acc']:.4f}")


if __name__ == "__main__":
    main()
