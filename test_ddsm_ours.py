"""Evaluate a trained double-view GMIC checkpoint."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from GMIC.src.modeling.double_gmic import GMIC
from multi_dataset_ddsm import create_dataloaders
from train_ddsm_ours import PROJECT_ROOT, build_device, build_model_parameters, evaluate, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "ddsm_csv" / "newtest1.csv")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "ddsm_images" / "all_imgs")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "result_ckp" / "best_model.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
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
    return parser.parse_args()


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = build_device(args.device)

    dataloaders = create_dataloaders(
        train_csv=None,
        valid_csv=None,
        test_csv=str(args.test_csv),
        images_dir=str(args.images_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
    )

    model_args = SimpleNamespace(**vars(args))
    model = GMIC(build_model_parameters(model_args, device)).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint, device))

    result = evaluate(model, dataloaders["test"], device, "test", args.num_classes)
    print(
        f"Test: loss={result['loss']:.4f} "
        f"auc={result['macro_auc']:.4f} "
        f"f1={result['macro_f1']:.4f} "
        f"acc={result['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
