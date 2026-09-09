#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Export the supplied trusted legacy FaceV2-n checkpoint once, outside ROS runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


UPSTREAM_REVISION = '956be8e642b5c10af4a1533e09084ca32ff4f21f'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--image', type=Path, help='Also compare on a real camera frame')
    args = parser.parse_args()
    weights = args.weights.expanduser().resolve(strict=True)
    upstream = args.upstream_dir.expanduser().resolve(strict=True)
    revision = subprocess.check_output(
        ['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f'Expected YOLOv5 v6.0 source at {UPSTREAM_REVISION}')
    sys.path.insert(0, str(upstream))
    # Upstream's loader predates PyTorch's weights_only default. This tool is for
    # the user-supplied trusted checkpoint, not arbitrary downloaded pickles.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from models.experimental import attempt_load

    torch.set_num_threads(2)
    model = attempt_load(str(weights), map_location='cpu', fuse=False)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Upsample):
            layer.recompute_scale_factor = None

    class OutputOnly(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            return self.network(image)[0]

    network = OutputOnly(model).eval()
    output = weights.with_suffix('.onnx')
    temporary = output.with_suffix('.part.onnx')
    sample = torch.zeros(1, 3, 640, 640)
    with torch.inference_mode():
        network(sample)
        torch.onnx.export(
            network, sample, str(temporary), input_names=['images'],
            output_names=['output'], opset_version=12, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(temporary), sess_options=options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(7)
    samples = [sample.numpy(), rng.random((1, 3, 640, 640), dtype=np.float32)]
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f'Cannot read {args.image}')
        image = cv2.resize(image, (640, 640))[:, :, ::-1].transpose(2, 0, 1)
        samples.append(np.ascontiguousarray(image)[None].astype(np.float32) / 255)
    max_error = 0.0
    for image in samples:
        with torch.inference_mode():
            expected = network(torch.from_numpy(image)).numpy()
        actual = session.run(None, {'images': image})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    temporary.replace(output)
    manifest = {
        'source_sha256': sha256(weights), 'onnx_sha256': sha256(output),
        'upstream': f'https://github.com/ultralytics/yolov5/tree/{revision}',
        'torch': torch.__version__, 'onnxruntime': ort.__version__,
        'comparisons': len(samples), 'max_absolute_error': max_error,
    }
    output.with_suffix('.export.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Export verified: {output}; max absolute difference {max_error:.6g}')


if __name__ == '__main__':
    main()
