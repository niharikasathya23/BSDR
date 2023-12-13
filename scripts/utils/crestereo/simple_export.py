import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import blobconverter
import yaml
from copy import deepcopy

from nets import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', '--output_dir', type=str, help="Path to output directory", default="output")
    args = parser.parse_args()

    device = torch.device('cpu')
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 384, 640)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_name = "crestereo_384x640"

    print("Converting PyTorch model to ONNX")
    # ONNX export step
    torch.onnx.export(
        model,
        (dummy_input, dummy_input),
        f"{args.output_dir}/{model_name}.onnx",
        # input_names=['right', 'disp'],
        opset_version=12,
        verbose=True
    )

    print("Converting ONNX to IR")
    os.system(f"mo \
    --input_model {args.output_dir}/{model_name}.onnx \
    --model_name {model_name} \
    --data_type FP16 \
    --output_dir {args.output_dir}/")

    print("Converting IR to blob")
    xmlfile = f"{args.output_dir}/{model_name}.xml"
    binfile = f"{args.output_dir}/{model_name}.bin"
    blob_path = blobconverter.from_openvino(
        xml=xmlfile,
        bin=binfile,
        data_type="FP16",
        shaves=6,
        output_dir=args.output_dir
    )
