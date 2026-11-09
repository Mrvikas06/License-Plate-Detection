"""
License Plate Detection - Inference Script (predict.py)
Detects license plates in images, videos, directories, or live webcam streams.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO


CANDIDATE_WEIGHTS = [
    os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    os.path.join(os.path.dirname(__file__), "license-plate-detection", "runs", "detect", "train-2", "weights", "best.pt")
]

DEFAULT_WEIGHTS = next((w for w in CANDIDATE_WEIGHTS if os.path.exists(w)), CANDIDATE_WEIGHTS[0])


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 License Plate Detection - Inference")
    parser.add_argument(
        "--source",
        type=str,
        default="license-plate-detection/test/images",
        help="Path to image, video, directory, or webcam index (e.g. 0)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Path to trained model weights (.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detection (0.0 to 1.0)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold"
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
        default="",
        help="Device to run on (e.g. '0', 'cpu', '')"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated detection results"
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        default=False,
        help="Save cropped detected license plates"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Project directory to save results"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predict",
        help="Experiment run name"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display results interactively on screen"
    )
    return parser.parse_args()


def run_inference(
    source,
    weights=DEFAULT_WEIGHTS,
    conf=0.35,
    iou=0.45,
    imgsz=640,
    device="",
    save=True,
    save_crop=False,
    project="runs",
    name="predict",
    show=False,
):
    """Executes license plate detection inference using YOLO11."""
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found at: {weights}")

    print(f"Loading YOLO11 License Plate Detection Model from: {weights}")
    model = YOLO(weights)

    # Convert numeric string to integer if webcam source is specified
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"Running inference on source: {source}")
    print(f"Confidence threshold: {conf} | IoU threshold: {iou} | Image size: {imgsz}")

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device if device else None,
        save=save,
        save_crop=save_crop,
        project=project,
        name=name,
        show=show,
    )

    total_detections = 0
    for r in results:
        total_detections += len(r.boxes)

    print(f"\nInference complete! Processed {len(results)} image(s)/frame(s).")
    print(f"Total license plates detected: {total_detections}")
    if save and len(results) > 0 and hasattr(results[0], "save_dir"):
        print(f"Results saved to: {os.path.abspath(results[0].save_dir)}")

    return results


def main():
    args = parse_args()
    run_inference(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        show=args.show,
    )


if __name__ == "__main__":
    main()
