from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import build_student
from src.utils import file_size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KD Student PyTorch model.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--student_model",
        type=str,
        default="mobilenet_v2",
    )

    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8],
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
    )

    return parser.parse_args()


def measure(model, batch, warmup, repeat):

    with torch.no_grad():

        for _ in range(warmup):
            _ = model(batch)

        times = []

        for _ in range(repeat):

            start = time.perf_counter()

            _ = model(batch)

            end = time.perf_counter()

            times.append((end - start) * 1000)

    return times


def main():

    args = parse_args()

    device = torch.device("cpu")

    model = build_student(
        num_classes=args.num_classes,
        student_model=args.student_model,
        pretrained=False,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    model.to(device)

    rows = []

    size_mb = file_size_mb(args.checkpoint)

    for bs in args.batch_sizes:

        batch = torch.randn(
            bs,
            3,
            args.img_size,
            args.img_size,
        ).to(device)

        times = measure(
            model,
            batch,
            args.warmup,
            args.repeat,
        )

        mean_latency = float(np.mean(times))

        rows.append(
            {
                "model": "kd_student",
                "checkpoint": args.checkpoint,
                "batch_size": bs,
                "mean_latency_ms": mean_latency,
                "median_latency_ms": float(np.median(times)),
                "p95_latency_ms": float(np.percentile(times, 95)),
                "throughput_img_per_sec": float(
                    bs / (mean_latency / 1000)
                ),
                "model_size_mb": size_mb,
            }
        )

    df = pd.DataFrame(rows)

    Path(args.output_csv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.output_csv,
        index=False,
    )

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()