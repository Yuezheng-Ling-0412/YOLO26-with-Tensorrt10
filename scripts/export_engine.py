#!/usr/bin/env python3
"""
YOLO26n pt → ONNX → TensorRT FP16 engine 转换脚本
用法:
    python3 scripts/export_engine.py \
        --weights models/best.pt \
        --output  models/best.engine \
        --imgsz   640
"""

import argparse
import os
import sys
from pathlib import Path

# 转换pt为onnx
def export_onnx(weights: str, output_onnx: str, imgsz: int) -> None:
    from ultralytics import YOLO
    model = YOLO(weights)
    model.export(
        format="onnx",
        imgsz=imgsz,    # 输出图像尺寸
        opset=11,
        simplify=True,  # 简化onnx图，去除冗余节点
        dynamic=False,  # 固定输入输出shape，trt使用静态shape
    )
    # ultralytics 默认在权重同目录生成 .onnx
    src = Path(weights).with_suffix(".onnx")
    if str(src) != output_onnx:
        import shutil
        shutil.move(str(src), output_onnx)
    print(f"[export] ONNX saved → {output_onnx}")

# 从onnx构建trt引擎，支持fp16
# fp16: 是否尝试启用 FP16 精度（若硬件支持）
def build_engine(onnx_path: str, engine_path: str, fp16: bool = True) -> None:
    import tensorrt as trt
    # 创建日志
    logger = trt.Logger(trt.Logger.INFO)
    # 创建构建引擎
    builder = trt.Builder(logger)
    # 创建网络定义，显式设置批处理标志
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    # onnx解析器
    parser = trt.OnnxParser(network, logger)
    # 读取onnx并解析
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            # 解析失败打印错误信息
            for i in range(parser.num_errors):
                print(f"[TRT] parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")
    # 创建构建配置用于指定精度/内存池大小等
    config = builder.create_builder_config()
    # 限制工作空间临时内存1GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # 如果要求fp16且硬件支持，启用fp16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled")
    else:
        print("[build] FP16 not available, using FP32")
    # 构建序列化引擎（实际进行网络优化、层融合、内核生成等）
    print("[build] Building TensorRT engine (may take a few minutes) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build] Engine saved → {engine_path}")

    # 打印输入输出信息
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(serialized)
    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode  = engine.get_tensor_mode(name)
        print(f"  {'INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'} "
              f"{name:20s}  shape={tuple(shape)}  dtype={dtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo/runs/detect/train3/weights/best.pt")
    parser.add_argument("--output",  default="models/best.engine")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    onnx_path = args.output.replace(".engine", ".onnx")

    # Step1: pt → ONNX
    if not os.path.exists(onnx_path):
        export_onnx(args.weights, onnx_path, args.imgsz)
    else:
        print(f"[export] ONNX already exists: {onnx_path}")

    # Step2: ONNX → TRT engine
    build_engine(onnx_path, args.output, fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
