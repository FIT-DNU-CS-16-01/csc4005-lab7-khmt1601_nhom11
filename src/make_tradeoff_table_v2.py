from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create trade-off table for multiple models."
    )

    parser.add_argument(
        "--evals",
        nargs="+",
        required=True,
        help="Evaluation json files.",
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="Benchmark csv files.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names in benchmark csv.",
    )

    parser.add_argument(
        "--display_names",
        nargs="+",
        required=True,
        help="Names shown in output table.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_md",
        type=str,
        required=True,
    )

    return parser.parse_args()


def pick_latency(csv_path, model_name, batch_size):

    df = pd.read_csv(csv_path)

    row = df[
        (df["model"] == model_name)
        &
        (df["batch_size"] == batch_size)
    ].iloc[0]

    return (
        float(row["mean_latency_ms"]),
        float(row["throughput_img_per_sec"]),
        float(row["model_size_mb"]),
    )


def make_comment(size):

    if size > 200:
        return "Độ chính xác cao nhưng mô hình lớn, tốc độ thấp."

    elif size > 20:
        return "Giảm kích thước đáng kể, tăng tốc suy luận, phù hợp CPU."

    else:
        return "Mô hình rất nhỏ, throughput cao, phù hợp triển khai edge/Smart Campus."


def main():

    args = parse_args()

    if not (
        len(args.evals)
        == len(args.benchmarks)
        == len(args.models)
        == len(args.display_names)
    ):
        raise ValueError(
            "evals, benchmarks, models và display_names phải cùng số lượng."
        )

    rows = []

    for eval_file, bench_file, model_name, display_name in zip(
        args.evals,
        args.benchmarks,
        args.models,
        args.display_names,
    ):

        metrics = load_json(eval_file)

        latency, throughput, size = pick_latency(
            bench_file,
            model_name,
            args.batch_size,
        )

        rows.append(
            {
                "Model": display_name,
                "Accuracy": metrics.get("accuracy"),
                "Macro-F1": metrics.get("macro_f1"),
                "Mean latency @bs=1 (ms)": latency,
                "Throughput (img/s)": throughput,
                "Model size (MB)": size,
                "Nhận xét": make_comment(size),
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

    md = df.to_markdown(index=False)

    Path(args.output_md).write_text(
        md,
        encoding="utf-8",
    )

    print(md)


if __name__ == "__main__":
    main()