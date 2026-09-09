# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Image-only inference and coordinate conversion; no ROS or simulator state."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MODEL_FILES = {
    'yunet': 'face_detection_yunet_2023mar.onnx',
    'yolov5n_face': 'yolov5n-face.onnx',
    'yolo_facev2n': 'yolo-facev2n-preweight.onnx',
}


@dataclass(frozen=True)
class Face:
    """Axis-aligned box in original image pixels."""

    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2)


def letterbox(image, size=640):
    """Resize without stretching; return exact scale and integer padding."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    resized = cv2.resize(image, (new_width, new_height))
    padded = cv2.copyMakeBorder(
        resized, top, size - new_height - top, left, size - new_width - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, (new_width / width, new_height / height, left, top)


def decode_yolo(output, image_shape, transform, threshold, nms_threshold):
    """Decode supplied YOLO outputs: six columns or sixteen with landmarks."""
    rows = np.asarray(output)[0]
    if rows.ndim != 2 or rows.shape[1] not in (6, 16):
        raise ValueError(f'Unsupported face model output shape: {output.shape}')
    scores = rows[:, 4] * rows[:, -1]
    valid = np.isfinite(rows).all(axis=1) & (scores >= threshold)
    rows, scores = rows[valid], scores[valid]
    sx, sy, left, top = transform
    height, width = image_shape[:2]
    boxes = []
    confidences = []
    for row, score in zip(rows, scores):
        x1 = np.clip((row[0] - row[2] / 2 - left) / sx, 0, width)
        y1 = np.clip((row[1] - row[3] / 2 - top) / sy, 0, height)
        x2 = np.clip((row[0] + row[2] / 2 - left) / sx, 0, width)
        y2 = np.clip((row[1] + row[3] / 2 - top) / sy, 0, height)
        if x2 > x1 and y2 > y1:
            boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            confidences.append(float(score))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)
    return [Face(*boxes[i], confidences[i]) for i in np.asarray(indices).reshape(-1)]


class YuNetDetector:
    """Use OpenCV's maintained YuNet decoder, including landmark output."""

    def __init__(self, path, threshold, nms_threshold):
        self.model = cv2.FaceDetectorYN.create(
            str(path), '', (640, 480), threshold, nms_threshold, 5000)

    def detect(self, image):
        self.model.setInputSize((image.shape[1], image.shape[0]))
        _, rows = self.model.detect(image)
        if rows is None:
            return []
        return [Face(*(float(v) for v in row[:4]), float(row[-1])) for row in rows]


class YoloDetector:
    """Run a prepared ONNX model on CPU with a bounded number of threads."""

    def __init__(self, path, threshold, nms_threshold, threads):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(
            str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name
        self.threshold = threshold
        self.nms_threshold = nms_threshold

    def detect(self, image):
        padded, transform = letterbox(image)
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
        tensor = tensor.astype(np.float32)[None] / 255.0
        output = self.model.run(None, {self.input_name: tensor})[0]
        return decode_yolo(
            output, image.shape, transform, self.threshold, self.nms_threshold)


def create_detector(name, model_dir, threshold=0.6, nms_threshold=0.4, threads=2):
    """Load exactly one explicitly selected model; never download or substitute weights."""
    if name not in MODEL_FILES:
        raise ValueError(f'Unknown detector {name!r}; choose {list(MODEL_FILES)}')
    if not 0 < threshold < 1 or not 0 < nms_threshold < 1 or threads < 1:
        raise ValueError('Thresholds must be between 0 and 1, threads must be positive')
    path = Path(model_dir).expanduser() / MODEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f'Model missing: {path}. See docs/face_detection_test.md')
    cv2.setNumThreads(threads)
    if name == 'yunet':
        return YuNetDetector(path, threshold, nms_threshold)
    return YoloDetector(path, threshold, nms_threshold, threads)
