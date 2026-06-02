import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from GMIC.src.modeling.double_gmic import GMIC
from multi_dataset_ddsm import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "ddsm_csv" / "newtrain1.csv")
    parser.add_argument("--valid-csv", type=Path, default=PROJECT_ROOT / "ddsm_csv" / "newvalid1.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "ddsm_csv" / "newtest1.csv")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "ddsm_images" / "all_imgs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "result_ckp")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-8)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--image-size", type=int, nargs=2, default=(1024, 512), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--crop-shape", type=int, nargs=2, default=(256, 256), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--cam-size", type=int, nargs=2, default=(46, 30), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--post-processing-dim", type=int, default=256)
    parser.add_argument("--num-crops", type=int, default=6)

    parser.add_argument("--ofu-in-channels", type=int, default=256)
    parser.add_argument("--ofu-out-channels", type=int, default=256)
    parser.add_argument("--ofu-scale", type=int, default=2)
    parser.add_argument("--ofu-grid", type=str, default="geo")
    parser.add_argument("--ofu-norm", type=str, default="bn")
    parser.add_argument("--ofu-act", type=str, default="gelu")

    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--kan-grid-size", type=int, default=5)
    parser.add_argument("--kan-spline-order", type=int, default=3)
    parser.add_argument("--eval-split", choices=("valid", "test"), default="valid")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model_parameters(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    gpu_number = 0
    if device.type == "cuda":
        gpu_number = device.index if device.index is not None else torch.cuda.current_device()

    return {
        "cam_size": tuple(args.cam_size),
        "crop_shape": tuple(args.crop_shape),
        "num_classes": args.num_classes,
        "post_processing_dim": args.post_processing_dim,
        "device_type": "gpu" if device.type == "cuda" else "cpu",
        "gpu_number": gpu_number,
        "K": args.num_crops,
        "ofu_in_channels": args.ofu_in_channels,
        "ofu_out_channels": args.ofu_out_channels,
        "ofu_scale": args.ofu_scale,
        "ofu_grid": args.ofu_grid,
        "ofu_norm": args.ofu_norm,
        "ofu_act": args.ofu_act,
        "feature_dropout": args.feature_dropout,
        "fusion_dropout": args.fusion_dropout,
        "kan_grid_size": args.kan_grid_size,
        "kan_spline_order": args.kan_spline_order,
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_loss(outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    output_fusion, output_global, output_local = outputs
    return (
        F.cross_entropy(output_fusion, target)
        + F.cross_entropy(output_global, target)
        + F.cross_entropy(output_local, target)
    )


def np_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=axis, keepdims=True)


def safe_mean(values: Iterable[float]) -> float:
    valid_values = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid_values)) if valid_values else float("nan")


def summarize_predictions(
    truth: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    probabilities = np_softmax(logits, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    truth_bin = label_binarize(truth, classes=np.arange(num_classes))

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "macro_auc": float("nan"),
        "weighted_auc": float("nan"),
        "mean_auprc": float("nan"),
        "per_class": [],
    }

    try:
        metrics["macro_auc"] = float(roc_auc_score(truth_bin, probabilities, average="macro"))
        metrics["weighted_auc"] = float(roc_auc_score(truth_bin, probabilities, average="weighted"))
    except ValueError:
        pass

    average_precision_scores = []
    for class_idx in range(num_classes):
        binary_truth = truth_bin[:, class_idx]
        class_scores = probabilities[:, class_idx]
        class_predictions = predictions == class_idx
        class_truth = truth == class_idx

        true_positive = int(np.sum(class_predictions & class_truth))
        false_negative = int(np.sum(~class_predictions & class_truth))
        true_negative = int(np.sum(~class_predictions & ~class_truth))
        false_positive = int(np.sum(class_predictions & ~class_truth))

        class_auc = float("nan")
        if len(np.unique(binary_truth)) == 2:
            fpr, tpr, _ = roc_curve(binary_truth, class_scores)
            class_auc = float(auc(fpr, tpr))
            average_precision_scores.append(float(average_precision_score(binary_truth, class_scores)))

        metrics["per_class"].append(
            {
                "class": class_idx,
                "auc": class_auc,
                "sensitivity": true_positive / (true_positive + false_negative + 1e-9),
                "specificity": true_negative / (true_negative + false_positive + 1e-9),
            }
        )

    metrics["mean_auprc"] = safe_mean(average_precision_scores)
    return metrics


def train_one_epoch(
    model: GMIC,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    num_classes: int,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    truth, logits = [], []

    progress = tqdm(loader, total=len(loader), desc=f"train {epoch}/{epochs}")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        target = batch["cancer"].long()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = compute_loss(outputs, target)
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        truth.append(target.detach().cpu().numpy())
        logits.append(outputs[0].detach().cpu().numpy())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    truth_np = np.concatenate(truth)
    logits_np = np.concatenate(logits)
    result = summarize_predictions(truth_np, logits_np, num_classes)
    result["loss"] = total_loss / max(total_samples, 1)
    return result


@torch.no_grad()
def evaluate(
    model: GMIC,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split: str,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    truth, logits = [], []

    progress = tqdm(loader, total=len(loader), desc=split)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        target = batch["cancer"].long()
        outputs = model(batch)
        loss = compute_loss(outputs, target)

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        truth.append(target.cpu().numpy())
        logits.append(outputs[0].cpu().numpy())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    truth_np = np.concatenate(truth)
    logits_np = np.concatenate(logits)
    result = summarize_predictions(truth_np, logits_np, num_classes)
    result["loss"] = total_loss / max(total_samples, 1)
    return result


class EarlyStopping:
    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return False

        self.counter += 1
        return self.counter >= self.patience


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max((epoch + 1) / warmup_epochs, min_lr / base_lr)

        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(cosine_factor, min_lr / base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def print_epoch(epoch: int, train_result: Dict[str, Any], eval_result: Dict[str, Any], lr: float) -> None:
    print(
        f"Epoch {epoch}: "
        f"lr={lr:.3e} "
        f"train_loss={train_result['loss']:.4f} "
        f"eval_loss={eval_result['loss']:.4f} "
        f"eval_auc={eval_result['macro_auc']:.4f} "
        f"eval_f1={eval_result['macro_f1']:.4f} "
        f"eval_acc={eval_result['accuracy']:.4f}"
    )

    for class_result in eval_result["per_class"]:
        print(
            f"  class {class_result['class']}: "
            f"auc={class_result['auc']:.4f} "
            f"sens={class_result['sensitivity']:.4f} "
            f"spec={class_result['specificity']:.4f}"
        )


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def save_history(history: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(history), handle, indent=2)

    epochs = list(range(1, len(history["train"]) + 1))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["lr"], marker="o")
    plt.title("Learning rate")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [item["loss"] for item in history["train"]], marker="o", label="train")
    plt.plot(epochs, [item["loss"] for item in history["eval"]], marker="o", label="eval")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = build_device(args.device)
    dataloaders = create_dataloaders(
        train_csv=str(args.train_csv),
        valid_csv=str(args.valid_csv) if args.valid_csv else None,
        test_csv=str(args.test_csv) if args.test_csv else None,
        images_dir=str(args.images_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
    )

    eval_loader = dataloaders.get(args.eval_split)
    if eval_loader is None:
        raise ValueError(f"Requested eval split {args.eval_split!r} was not created.")

    model = GMIC(build_model_parameters(args, device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr, args.lr)
    early_stopping = EarlyStopping(args.patience, args.output_dir / "best_model.pth")

    history: Dict[str, Any] = {"train": [], "eval": [], "lr": []}
    best_macro_auc = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            epoch,
            args.epochs,
            args.num_classes,
        )
        eval_result = evaluate(model, eval_loader, device, args.eval_split, args.num_classes)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train"].append(train_result)
        history["eval"].append(eval_result)
        history["lr"].append(current_lr)

        if not math.isnan(eval_result["macro_auc"]):
            best_macro_auc = max(best_macro_auc, eval_result["macro_auc"])

        print_epoch(epoch, train_result, eval_result, current_lr)
        print(f"Best {args.eval_split} macro AUC so far: {best_macro_auc:.4f}")

        save_history(history, args.output_dir)
        if early_stopping.step(eval_result["loss"], model):
            print(f"Early stopping after {epoch} epochs.")
            break

    torch.save(model.state_dict(), args.output_dir / "last_model.pth")

    if "test" in dataloaders and args.eval_split != "test":
        best_checkpoint = args.output_dir / "best_model.pth"
        if best_checkpoint.exists():
            model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        test_result = evaluate(model, dataloaders["test"], device, "test", args.num_classes)
        history["test"] = test_result
        save_history(history, args.output_dir)
        print(
            f"Test: loss={test_result['loss']:.4f} "
            f"auc={test_result['macro_auc']:.4f} "
            f"f1={test_result['macro_f1']:.4f} "
            f"acc={test_result['accuracy']:.4f}"
        )

    print(f"Training complete. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
