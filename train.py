"""
License Plate Detection - Training Script (train.py)
Reproducible training pipeline for YOLO11 Nano on the license plate dataset.
"""

import argparse
import os
from ultralytics import YOLO


CANDIDATE_DATA = [
    os.path.join(os.path.dirname(__file__), "dataset.yaml"),
    os.path.join(os.path.dirname(__file__), "license-plate-detection", "data.yaml")
]
DEFAULT_DATA = next((d for d in CANDIDATE_DATA if os.path.exists(d)), CANDIDATE_DATA[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 License Plate Detection Model")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="Path to data.yaml dataset descriptor"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model architecture or checkpoint to start from"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image resolution"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (lr0 * lrf)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience (epochs with no improvement)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu'"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker threads"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save experiment runs"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment run name"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from most recent checkpoint"
    )
    return parser.parse_args()


def train_model(
    data=DEFAULT_DATA,
    model_name="yolo11n.pt",
    epochs=50,
    batch=8,
    imgsz=640,
    lr0=0.01,
    lrf=0.01,
    patience=100,
    device="0",
    workers=8,
    project="runs/detect",
    name="train",
    resume=False,
):
    """Executes YOLO11 training workflow."""
    print("=" * 60)
    print("        STARTING YOLO11 LICENSE PLATE TRAINING PIPELINE")
    print("=" * 60)
    print(f"Base Model : {model_name}")
    print(f"Dataset    : {data}")
    print(f"Epochs     : {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    print(f"Device     : {device} | Workers: {workers}")
    print("=" * 60)

    model = YOLO(model_name)

    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        device=device,
        workers=workers,
        project=project,
        name=name,
        resume=resume,
        pretrained=True,
        plots=True,
    )

    print("\nTraining completed successfully!")
    print(f"Weights and logs saved to: {os.path.join(project, name)}")
    return results


def main():
    args = parse_args()
    train_model(
        data=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
