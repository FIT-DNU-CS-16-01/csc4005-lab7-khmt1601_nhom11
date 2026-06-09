from __future__ import annotations

import argparse
from pathlib import Path
from onnx import TensorProto

from onnxruntime.quantization import QuantType, quantize_dynamic

from src.utils import file_size_mb, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Apply dynamic quantization to ONNX model.")
    parser.add_argument("--input_onnx", type=str, required=True)
    parser.add_argument("--output_onnx", type=str, required=True)
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic"])
    parser.add_argument("--weight_type", type=str, default="QInt8", choices=["QInt8", "QUInt8"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_onnx)
    output_path = Path(args.output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX not found: {input_path}")

    weight_type = QuantType.QInt8 if args.weight_type == "QInt8" else QuantType.QUInt8

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),

        weight_type=weight_type,

        # 🔥 QUAN TRỌNG NHẤT: chỉ quantize linear layers
        op_types_to_quantize=["MatMul", "Gemm"],

        # ❌ không exclude conv kiểu hack nữa
        nodes_to_exclude=None,

        extra_options={
            "DefaultTensorType": TensorProto.FLOAT,

            # 🔥 CHỐT: tránh ConvInteger graph
            "MatMulConstBOnly": True,

            # 🔥 giữ graph stable
            "EnableSubgraph": False,

            # 🔥 tránh ORT strict check lỗi metadata
            "ForceQuantizeNoInputCheck": True
        }
    )

    report = {
        "method": "onnx_dynamic_quantization",
        "input_onnx": str(input_path),
        "output_onnx": str(output_path),
        "weight_type": args.weight_type,
        "baseline_size_mb": file_size_mb(input_path),
        "compressed_size_mb": file_size_mb(output_path),
        "size_reduction_percent": (1 - file_size_mb(output_path) / file_size_mb(input_path)) * 100,
    }
    save_json(report, output_path.parent / "quantization_report.json")
    print(report)


if __name__ == "__main__":
    main()
