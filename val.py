"""
License Plate Detection - Evaluation Script (val.py)
Evaluates trained model weights on validation or test dataset splits.
Reports Precision, Recall, mAP@50, mAP@50:95, and inference speed.
"""

import argparse
import os
import sys
from ultralytics import YOLO


CANDIDATE_WEIGHTS = [
    os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    os.path.join(os.path.dirname(__file__), "license-plate-detection", "runs", "detect", "train-2", "weights", "best.pt")
]
DEFAULT_WEIGHTS = next((w for w in CANDIDATE_WEIGHTS if os.path.exists(w)), CANDIDATE_WEIGHTS[0])

CANDIDATE_DATA = [
    os.path.join(os.path.dirname(__file__), "dataset.yaml"),
    os.path.join(os.path.dirname(__file__), "license-plate-detection", "data.yaml")
]
DEFAULT_DATA = next((d for d in CANDIDATE_DATA if os.path.exists(d)), CANDIDATE_DATA[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 License Plate Detection Model")
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Path to trained model weights (.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="Path to data.yaml dataset descriptor"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on ('val', 'test', or 'train')"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device (e.g. '0', 'cpu')"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Directory to save evaluation artifacts"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="eval",
        help="Evaluation run name"
    )
    return parser.parse_args()


def evaluate_model(
    weights=DEFAULT_WEIGHTS,
    data=DEFAULT_DATA,
    split="test",
    batch=16,
    imgsz=640,
    device="0",
    project="runs",
    name="eval",
):
    """Executes model validation/testing and prints formatted metrics."""
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found at: {weights}")
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data configuration not found at: {data}")

    print(f"Loading weights: {weights}")
    print(f"Dataset config: {data} | Split: {split} | Batch: {batch} | Imgsz: {imgsz}")

    model = YOLO(weights)

    metrics = model.val(
        data=data,
        split=split,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        plots=True,
    )

    mp = metrics.box.mp
    mr = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    speed = metrics.speed

    print("\n" + "=" * 55)
    print(f"       EVALUATION METRICS [{split.upper()} SPLIT]")
    print("=" * 55)
    print(f" Precision (P)      : {mp * 100:.2f}% ({mp:.4f})")
    print(f" Recall (R)         : {mr * 100:.2f}% ({mr:.4f})")
    print(f" mAP @ 0.50         : {map50 * 100:.2f}% ({map50:.4f})")
    print(f" mAP @ 0.50:0.95    : {map50_95 * 100:.2f}% ({map50_95:.4f})")
    print("-" * 55)
    print(" LATENCY BREAKDOWN (per image):")
    print(f" - Preprocess       : {speed.get('preprocess', 0):.2f} ms")
    print(f" - Inference        : {speed.get('inference', 0):.2f} ms")
    print(f" - Postprocess      : {speed.get('postprocess', 0):.2f} ms")
    total_ms = sum([speed.get('preprocess', 0), speed.get('inference', 0), speed.get('postprocess', 0)])
    print(f" - Total per image  : {total_ms:.2f} ms (~{1000.0 / total_ms if total_ms > 0 else 0:.1f} FPS)")
    print("=" * 55)
    print(f"Plots and metrics saved to: {os.path.join(project, name)}\n")

    return metrics


def main():
    args = parse_args()
    evaluate_model(
        weights=args.weights,
        data=args.data,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
