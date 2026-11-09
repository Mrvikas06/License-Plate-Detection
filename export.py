"""
License Plate Detection - Model Export Script (export.py)
Exports the trained PyTorch YOLO11 model weights (.pt) to deployable formats:
ONNX, TorchScript, OpenVINO, Engine (TensorRT), etc.
"""

import argparse
import os
from ultralytics import YOLO


CANDIDATE_WEIGHTS = [
    os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    os.path.join(os.path.dirname(__file__), "license-plate-detection", "runs", "detect", "train-2", "weights", "best.pt")
]
DEFAULT_WEIGHTS = next((w for w in CANDIDATE_WEIGHTS if os.path.exists(w)), CANDIDATE_WEIGHTS[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Export Trained YOLO11 License Plate Model")
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Path to trained PyTorch weights (.pt)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="torchscript",
        choices=["torchscript", "onnx", "openvino", "engine", "coreml", "tflite"],
        help="Target export format"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="FP16 half-precision export"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Dynamic input axes support"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify model graph (ONNX)"
    )
    return parser.parse_args()


def export_model(
    weights=DEFAULT_WEIGHTS,
    fmt="torchscript",
    imgsz=640,
    half=False,
    dynamic=False,
    simplify=True,
):
    """Exports model to target format."""
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found at: {weights}")

    print(f"Loading weights: {weights}")
    print(f"Export format: {fmt} | Image size: {imgsz} | Half precision: {half}")

    model = YOLO(weights)
    exported_path = model.export(
        format=fmt,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
    )

    print("\nExport completed successfully!")
    print(f"Exported artifact: {exported_path}")
    return exported_path


def main():
    args = parse_args()
    export_model(
        weights=args.weights,
        fmt=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )


if __name__ == "__main__":
    main()
