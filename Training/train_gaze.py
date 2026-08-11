#!/usr/bin/env python3
"""train and convert the offline Aspectus appearance-based gaze estimator"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import coremltools as ct
import numpy as np
import onnx
from onnx import numpy_helper
from onnx2torch import convert as convert_onnx
from PIL import Image, ImageEnhance, ImageStat
import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 60
HEAD_POSE_SCALE_DEGREES = 30.0
OPENVINO_MODEL_SHA256 = "f8f13707401547c7c5e146ba22da1bd6ada00f47ebf347be6d97c5acfaf4c2bd"
MINIMUM_FACE_CONFIDENCE = 0.70
MINIMUM_OPENNESS = 0.40
MAXIMUM_ABSOLUTE_HEAD_ROLL_DEGREES = 20.0
MINIMUM_VALIDATION_SAMPLES_PER_POSE = 100
MINIMUM_VALIDATION_LENS_SAMPLES = 50
MINIMUM_VALIDATION_LENS_SAMPLES_PER_POSE = 6
MINIMUM_VALIDATION_LENS_TARGETS_PER_POSE = 2
MINIMUM_VALIDATION_HORIZONTAL_POSE_SEPARATION_DEGREES = 6.0
MINIMUM_VALIDATION_VERTICAL_POSE_SEPARATION_DEGREES = 5.0
CANONICAL_MAXIMUM_MEDIAN_DEGREES = 2.0
CANONICAL_MAXIMUM_P95_DEGREES = 5.0
CANONICAL_MAXIMUM_LENS_P95_DEGREES = 3.0
CHECKPOINT_FORMAT_VERSION = 4
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = frozenset({1, 2, 3, CHECKPOINT_FORMAT_VERSION})
FINAL_EVALUATION_FILENAME = "final-evaluation.json"
CANDIDATE_MANIFEST_FILENAME = "candidate-manifest.json"
CANDIDATE_MANIFEST_FORMAT_VERSION = 2
ROBUSTNESS_SEEDS = (7, 19, 43)
PRIMARY_CANDIDATE_SEED = ROBUSTNESS_SEEDS[0]
FINAL_EVALUATION_LEDGER_FILENAME = ".aspectus-final-evaluation-ledger.json"
FINAL_EVALUATION_LEDGER_LOCK_FILENAME = ".aspectus-final-evaluation-ledger.lock"
MAXIMUM_COREML_ANGULAR_DIFFERENCE_DEGREES = 0.25
PRETRAINED_ARCHITECTURE = "openvino-gaze-estimation-adas-0002-personalized-symmetric-v3"
NATIVE_ARCHITECTURE = "shared-eye-spatial-cnn-head-pose-v2"
NATIVE_PARAMETER_COUNT = 156_226
LEGACY_INPUT_CONTRACT = "legacy-axis-aligned-eye-crops-v1"
CANONICAL_INPUT_CONTRACT = "canonical-paired-eye-v1"
CANONICAL_CROP_CONTRACT = {
    "version": 1,
    "coordinateSpace": "source-image-fraction-top-left",
    "axisExtractor": "farthest-contour-pair-ordered-image-x",
    "alignment": "circular-mean-paired-eye-axes",
    "center": "per-eye-axis-midpoint",
    "scale": "1.8x-maximum-eye-axis-length-pixels",
    "outputWidth": IMAGE_SIZE,
    "outputHeight": IMAGE_SIZE,
    "sampling": "core-image-affine-hq-downsample-edge-clamp",
    "colorSpace": "sRGB",
}
CANONICAL_LABEL_CONTRACT = {
    "version": 1,
    "units": "degrees",
    "origin": "physical-lens",
    "yawPositive": "subject-right",
    "pitchPositive": "up",
}
CANONICAL_HEAD_POSE_CONTRACT = {
    "version": 1,
    "source": "Vision.VNFaceObservation.face-rectangles-revision-3",
    "units": "degrees",
    "order": "yaw-pitch-roll",
    "yawPositive": "counterclockwise",
    "pitchPositive": "head-down",
    "rollPositive": "counterclockwise",
}
INPUT_PREPROCESSING_CONTRACTS = {
    LEGACY_INPUT_CONTRACT: {
        "contract": LEGACY_INPUT_CONTRACT,
        "image_width": IMAGE_SIZE,
        "image_height": IMAGE_SIZE,
        "color_space": "sRGB",
        "pixel_normalization": "rgb-zero-to-one",
        "image_resize": "none",
        "alignment": "none",
        "recorder_crop_contract": "historical-unversioned",
        "head_pose_units": "degrees",
        "head_pose_order": "yaw-pitch-roll",
        "head_pose_tensor_encoding": "raw-degrees",
        "head_pose_source_contract": "historical-unversioned",
    },
    CANONICAL_INPUT_CONTRACT: {
        "contract": CANONICAL_INPUT_CONTRACT,
        "image_width": IMAGE_SIZE,
        "image_height": IMAGE_SIZE,
        "color_space": "sRGB",
        "pixel_normalization": "rgb-zero-to-one",
        "image_resize": "none",
        "alignment": "recorder-canonical-paired-eye-v1",
        "recorder_crop_contract": CANONICAL_CROP_CONTRACT,
        "head_pose_units": "degrees",
        "head_pose_order": "yaw-pitch-roll",
        "head_pose_tensor_encoding": "raw-degrees",
        "head_pose_source_contract": CANONICAL_HEAD_POSE_CONTRACT,
    },
}
SCHEMA_FOUR_CROP_SCALE = 1.8
SCHEMA_FOUR_SERIALIZATION_RESOLUTION = 1e-12
SCHEMA_FOUR_DERIVED_ANGLE_TOLERANCE_DEGREES = 1e-6
SCHEMA_FOUR_DERIVED_CLIPPING_TOLERANCE = 1e-8
SCHEMA_FOUR_ALLOWED_PUPIL_SOURCES = frozenset({
    "visionLandmark", "contourCentroid", "none",
})
POSE_PROMPTS = ("neutral", "turnLeft", "turnRight", "lookUp", "lookDown")
TARGETS_PER_POSE = 27
GRID_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
TRAINING_GRID_ORDER = (
    12, 0, 24, 4, 20, 6, 18, 8, 16, 2, 22, 10, 14,
    1, 23, 5, 19, 9, 15, 3, 21, 7, 17, 11, 13,
)
TARGET_LABEL_ABSOLUTE_TOLERANCE = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EVALUATION_INTERVAL = 1
DEFAULT_MINIMUM_LEARNING_RATE = 1e-6
PRETRAINED_FUSION_HEAD_PARAMETER_COUNT = 191_043
TAIL_LOSS_FRACTION = 0.20
TAIL_LOSS_WEIGHT = 0.25
AUGMENTATION_PROFILES = {
    "baseline": {
        "brightness_range": [0.85, 1.15],
        "contrast_range": [0.85, 1.15],
        "scale_range": [0.94, 1.06],
        "translation_pixels": [-2.5, 2.5],
        "head_pose_noise_standard_deviation_degrees": 1.0,
        "paired_horizontal_mirror_probability": 0.5,
    },
    "strong": {
        "brightness_range": [0.80, 1.20],
        "contrast_range": [0.80, 1.20],
        "scale_range": [0.92, 1.08],
        "translation_pixels": [-3.5, 3.5],
        "head_pose_noise_standard_deviation_degrees": 1.5,
        "paired_horizontal_mirror_probability": 0.5,
    },
}
CANDIDATE_FACTOR_FIELDS = {
    "model": (
        "architecture", "output_contract", "pretrained_model_sha256",
        "fine_tuning", "trainable_parameters",
    ),
    "optimizer": ("optimizer",),
    "loss": ("loss",),
    "augmentation": ("augmentation",),
    "input": ("input_preprocessing",),
    "filtering": (
        "filtering", "training_data_sha256", "development_data_sha256",
        "training_samples", "development_samples",
    ),
    "training": ("batch_size", "training_epochs", "evaluation_interval"),
}
CANDIDATE_SHARED_FIELDS = (
    "training_sessions", "development_sessions", "source_training_data_sha256",
    "source_development_data_sha256", "setup_sha256", "known_completed_sessions",
    "source_training_samples", "source_development_samples",
    "software_versions", "trainer_sha256", "label_contract_sha256",
)
REQUIRED_COLUMNS = {
    "schema_version",
    "participant_id",
    "session_id",
    "split",
    "sample",
    "target_id",
    "target_kind",
    "target_x",
    "target_y",
    "target_yaw_deg",
    "target_pitch_deg",
    "pose_prompt",
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
    "face_conf",
    "open_l",
    "open_r",
    "left_image",
    "right_image",
}
SCHEMA_FOUR_COLUMNS = {
    "frame_id",
    "elapsed_s",
    "contour_points_l",
    "contour_points_r",
    "pupil_source_l",
    "pupil_source_r",
    "pupil_points_l",
    "pupil_points_r",
    "axis_start_x_l",
    "axis_start_y_l",
    "axis_end_x_l",
    "axis_end_y_l",
    "axis_start_x_r",
    "axis_start_y_r",
    "axis_end_x_r",
    "axis_end_y_r",
    "alignment_rotation_deg",
    "alignment_disagreement_deg",
    "crop_side_px",
    "crop_clipped_fraction_l",
    "crop_clipped_fraction_r",
}


@dataclass(frozen=True)
class EyeQualityEvidence:
    contour_point_count: int
    pupil_source: str
    pupil_point_count: int
    axis_start: tuple[float, float]
    axis_end: tuple[float, float]
    crop_clipped_fraction: float


@dataclass(frozen=True)
class CanonicalAlignmentEvidence:
    source_image_size: tuple[int, int]
    left: EyeQualityEvidence
    right: EyeQualityEvidence
    rotation_degrees: float
    disagreement_degrees: float
    crop_side_pixels: float


@dataclass(frozen=True)
class Sample:
    session_id: str
    split: str
    target_id: int
    target_kind: str
    left_path: Path
    right_path: Path
    head_pose: tuple[float, float, float]
    gaze_degrees: tuple[float, float]
    pose_prompt: str = ""
    target_position: tuple[float, float] = (0, 0)
    face_confidence: float = math.nan
    minimum_openness: float = math.nan
    schema_version: int = 0
    participant_binding: str = ""
    setup_binding: str = ""
    session_created_at: str = ""
    session_metadata_sha256: str = ""
    manifest_sha256: str = ""
    input_contract: str = ""
    label_contract_sha256: str = ""
    canonical_alignment: CanonicalAlignmentEvidence | None = None


@dataclass(frozen=True)
class Evaluation:
    count: int
    angular_median_degrees: float
    angular_p95_degrees: float
    lens_count: int
    lens_p95_degrees: float
    yaw_mae_degrees: float
    pitch_mae_degrees: float


@dataclass(frozen=True)
class QualityGates:
    maximum_median_degrees: float
    maximum_p95_degrees: float
    maximum_lens_p95_degrees: float


@dataclass(frozen=True)
class EpochEvaluation:
    epoch: int
    learning_rate: float
    session_evaluations: dict[str, Evaluation]
    selection_score: float
    worst_lens_ratio: float
    worst_p95_ratio: float
    worst_median_ratio: float


@dataclass
class TrainingOutcome:
    model: nn.Module
    selected_epoch: int
    selected_evaluation: Evaluation
    selected_session_evaluations: dict[str, Evaluation]
    evaluated_epochs: list[EpochEvaluation]


class EyeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 96),
            nn.ReLU(),
        )

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        return self.features(eye)


class GazeEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eye_encoder = EyeEncoder()
        self.regressor = nn.Sequential(
            nn.Linear(96 * 2 + 3, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        head_pose: torch.Tensor,
    ) -> torch.Tensor:
        left = self.eye_encoder(left_eye)
        right = self.eye_encoder(right_eye)
        pose = head_pose / HEAD_POSE_SCALE_DEGREES
        return self.regressor(torch.cat((left, right, pose), dim=1))


def openvino_gaze_model(path: Path) -> nn.Module:
    if file_digest(path) != OPENVINO_MODEL_SHA256:
        raise ValueError("pretrained OpenVINO model checksum mismatch")
    source = onnx.load(path)
    dynamic_shapes = {
        "const_fold_opt__213": np.array([0, -1], dtype=np.int64),
        "new_shape__155": np.array([-1, 515, 1, 1], dtype=np.int64),
        "new_shape__156": np.array([-1, 1, 1, 3], dtype=np.int64),
    }
    for initializer in source.graph.initializer:
        if initializer.name in dynamic_shapes:
            initializer.CopyFrom(
                numpy_helper.from_array(dynamic_shapes[initializer.name], initializer.name)
            )
    return convert_onnx(source)


def is_pretrained_fusion_head_parameter(name: str) -> bool:
    return (
        "conv2d_14/Conv2D" in name
        or "conv2d_15/Conv2D" in name
        or name.startswith("Identity.")
    )


def configure_pretrained_trainable_scope(
    model: nn.Module, scope: str
) -> list[str]:
    if scope not in {"all", "fusion-head"}:
        raise ValueError(f"unsupported pretrained trainable scope: {scope}")
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = scope == "all" or is_pretrained_fusion_head_parameter(name)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("pretrained trainable scope selected no parameters")
    if scope == "fusion-head":
        count = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        if count != PRETRAINED_FUSION_HEAD_PARAMETER_COUNT:
            raise RuntimeError("pinned pretrained fusion-head contract changed")
    return trainable_names


class GazeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]):
    def __init__(
        self,
        samples: list[Sample],
        augment: bool,
        augmentation_strength: str = "baseline",
    ) -> None:
        self.samples = samples
        self.augmentation = (
            augmentation_config(augmentation_strength) if augment else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        left = self._load_eye(sample.left_path)
        right = self._load_eye(sample.right_path)
        gaze_degrees = sample.gaze_degrees
        if self.augmentation is not None:
            brightness = random.uniform(*self.augmentation["brightness_range"])
            contrast = random.uniform(*self.augmentation["contrast_range"])
            scale = random.uniform(*self.augmentation["scale_range"])
            offset_x = random.uniform(*self.augmentation["translation_pixels"])
            offset_y = random.uniform(*self.augmentation["translation_pixels"])
            left = ImageEnhance.Contrast(ImageEnhance.Brightness(left).enhance(brightness)).enhance(contrast)
            right = ImageEnhance.Contrast(ImageEnhance.Brightness(right).enhance(brightness)).enhance(contrast)
            left = affine_eye(left, scale, offset_x, offset_y)
            right = affine_eye(right, scale, offset_x, offset_y)
            head_pose_noise = self.augmentation[
                "head_pose_noise_standard_deviation_degrees"
            ]
            head_pose = tuple(
                value + random.gauss(0, head_pose_noise)
                for value in sample.head_pose
            )
            if random.random() < self.augmentation["paired_horizontal_mirror_probability"]:
                left, right, head_pose, gaze_degrees = mirror_gaze_sample(
                    left, right, head_pose, gaze_degrees)
        else:
            head_pose = sample.head_pose
        return (
            image_tensor(left),
            image_tensor(right),
            torch.tensor(head_pose, dtype=torch.float32),
            torch.tensor(gaze_degrees, dtype=torch.float32),
            sample.target_kind == "lens",
        )

    @staticmethod
    def _load_eye(path: Path) -> Image.Image:
        with Image.open(path) as image:
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"expected {IMAGE_SIZE}x{IMAGE_SIZE} eye crop: {path}")
            return image.convert("RGB")


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def rgb_to_bgr(images: torch.Tensor) -> torch.Tensor:
    return images[:, [2, 1, 0], :, :]


def mirror_gaze_sample(
    left: Image.Image,
    right: Image.Image,
    head_pose: tuple[float, float, float],
    gaze_degrees: tuple[float, float],
) -> tuple[Image.Image, Image.Image, tuple[float, float, float], tuple[float, float]]:
    return (
        right.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        left.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        (-head_pose[0], head_pose[1], -head_pose[2]),
        (-gaze_degrees[0], gaze_degrees[1]),
    )


def affine_eye(image: Image.Image, scale: float,
               offset_x: float, offset_y: float) -> Image.Image:
    inverse = 1.0 / scale
    centre = (IMAGE_SIZE - 1) / 2
    transform = (
        inverse, 0, centre - inverse * (centre + offset_x),
        0, inverse, centre - inverse * (centre + offset_y),
    )
    fill = tuple(round(value) for value in ImageStat.Stat(image).mean)
    return image.transform(
        image.size, Image.Transform.AFFINE, transform,
        resample=Image.Resampling.BICUBIC, fillcolor=fill,
    )


def finite_float(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"manifest contains a non-finite {column}")
    return value


def finite_integer(row: dict[str, str], column: str, minimum: int = 0) -> int:
    value = row[column]
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"manifest contains an invalid {column}") from error
    if str(parsed) != value or parsed < minimum:
        raise ValueError(f"manifest contains an invalid {column}")
    return parsed


def validated_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"session {name} is invalid")
    return value


def label_contract_digest() -> str:
    return canonical_digest(CANONICAL_LABEL_CONTRACT)


def matches_canonical_contract(value: object, expected: object) -> bool:
    try:
        return canonical_digest(value) == canonical_digest(expected)
    except (TypeError, ValueError):
        return False


def input_preprocessing_contract(contract: str) -> dict[str, object]:
    try:
        return deepcopy(INPUT_PREPROCESSING_CONTRACTS[contract])
    except KeyError as error:
        raise ValueError(f"unsupported input preprocessing contract: {contract}") from error


def validated_input_preprocessing(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("frozen checkpoint input preprocessing is invalid")
    contract = value.get("contract")
    if not isinstance(contract, str):
        raise ValueError("frozen checkpoint input preprocessing is invalid")
    expected = input_preprocessing_contract(contract)
    if not matches_canonical_contract(value, expected):
        raise ValueError("frozen checkpoint input preprocessing contract mismatch")
    return expected


def sample_input_contract(sample: Sample) -> str:
    if sample.input_contract:
        return sample.input_contract
    return CANONICAL_INPUT_CONTRACT if sample.schema_version == 4 else LEGACY_INPUT_CONTRACT


def sample_label_contract_sha256(sample: Sample) -> str:
    return sample.label_contract_sha256 or label_contract_digest()


def require_single_input_preprocessing(*sample_groups: Iterable[Sample]) -> dict[str, object]:
    contracts = {
        sample_input_contract(sample)
        for samples in sample_groups
        for sample in samples
    }
    if len(contracts) != 1:
        raise ValueError("training and development sessions mix input preprocessing contracts")
    return input_preprocessing_contract(next(iter(contracts)))


def require_single_label_contract(*sample_groups: Iterable[Sample]) -> str:
    digests = {
        sample_label_contract_sha256(sample)
        for samples in sample_groups
        for sample in samples
    }
    if len(digests) != 1:
        raise ValueError("training and development sessions mix label contracts")
    digest = validated_sha256(next(iter(digests)), "label contract")
    if digest != label_contract_digest():
        raise ValueError("unsupported label contract")
    return digest


def schema_four_axis(
    start: tuple[float, float], end: tuple[float, float],
    source_width: int, source_height: int,
) -> tuple[tuple[float, float], float, float]:
    first = (start[0] * source_width, start[1] * source_height)
    second = (end[0] * source_width, end[1] * source_height)
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        raise ValueError("schema-4 eye alignment axis is degenerate")
    return (
        ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2),
        length,
        math.atan2(dy, dx),
    )


def clip_polygon(
    polygon: list[tuple[float, float]],
    inside: Callable[[tuple[float, float]], bool],
    intersection: Callable[
        [tuple[float, float], tuple[float, float]], tuple[float, float]
    ],
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    previous = polygon[-1]
    result: list[tuple[float, float]] = []
    for current in polygon:
        if inside(current):
            if not inside(previous):
                result.append(intersection(previous, current))
            result.append(current)
        elif inside(previous):
            result.append(intersection(previous, current))
        previous = current
    return result


def clipped_crop_fraction(
    center: tuple[float, float], rotation: float, side: float,
    source_width: int, source_height: int,
) -> float:
    half = side / 2
    axis = (math.cos(rotation), math.sin(rotation))
    perpendicular = (-axis[1], axis[0])
    polygon = [
        (
            center[0] + x * half * axis[0] + y * half * perpendicular[0],
            center[1] + x * half * axis[1] + y * half * perpendicular[1],
        )
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]

    def vertical(boundary: float):
        def intersection(first: tuple[float, float], second: tuple[float, float]):
            dx = second[0] - first[0]
            amount = (boundary - first[0]) / dx if abs(dx) > 1e-12 else 0
            return boundary, first[1] + amount * (second[1] - first[1])
        return intersection

    def horizontal(boundary: float):
        def intersection(first: tuple[float, float], second: tuple[float, float]):
            dy = second[1] - first[1]
            amount = (boundary - first[1]) / dy if abs(dy) > 1e-12 else 0
            return first[0] + amount * (second[0] - first[0]), boundary
        return intersection

    polygon = clip_polygon(polygon, lambda point: point[0] >= 0, vertical(0))
    polygon = clip_polygon(
        polygon, lambda point: point[0] <= source_width, vertical(source_width)
    )
    polygon = clip_polygon(polygon, lambda point: point[1] >= 0, horizontal(0))
    polygon = clip_polygon(
        polygon, lambda point: point[1] <= source_height, horizontal(source_height)
    )
    if len(polygon) < 3:
        visible_area = 0.0
    else:
        twice_area = sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
        visible_area = abs(twice_area) / 2
    return min(1.0, max(0.0, 1 - visible_area / (side * side)))


def schema_four_eye_evidence(
    row: dict[str, str], suffix: str,
) -> EyeQualityEvidence:
    contour_points = finite_integer(row, f"contour_points_{suffix}", minimum=2)
    pupil_source = row[f"pupil_source_{suffix}"]
    if pupil_source not in SCHEMA_FOUR_ALLOWED_PUPIL_SOURCES:
        raise ValueError("schema-4 pupil source is invalid")
    pupil_points = finite_integer(row, f"pupil_points_{suffix}")
    if (pupil_source == "visionLandmark") != (pupil_points > 0):
        raise ValueError("schema-4 pupil evidence is inconsistent")
    start = (
        finite_float(row, f"axis_start_x_{suffix}"),
        finite_float(row, f"axis_start_y_{suffix}"),
    )
    end = (
        finite_float(row, f"axis_end_x_{suffix}"),
        finite_float(row, f"axis_end_y_{suffix}"),
    )
    if not (start[0] < end[0] or (start[0] == end[0] and start[1] <= end[1])):
        raise ValueError("schema-4 eye alignment axis ordering is invalid")
    clipped_fraction = finite_float(row, f"crop_clipped_fraction_{suffix}")
    if not 0 <= clipped_fraction <= 1:
        raise ValueError("schema-4 crop clipping fraction is invalid")
    return EyeQualityEvidence(
        contour_point_count=contour_points,
        pupil_source=pupil_source,
        pupil_point_count=pupil_points,
        axis_start=start,
        axis_end=end,
        crop_clipped_fraction=clipped_fraction,
    )


def schema_four_alignment_evidence(
    row: dict[str, str], source_width: int, source_height: int,
) -> CanonicalAlignmentEvidence:
    left = schema_four_eye_evidence(row, "l")
    right = schema_four_eye_evidence(row, "r")
    left_center, left_length, left_angle = schema_four_axis(
        left.axis_start, left.axis_end, source_width, source_height
    )
    right_center, right_length, right_angle = schema_four_axis(
        right.axis_start, right.axis_end, source_width, source_height
    )
    sine = math.sin(left_angle) + math.sin(right_angle)
    cosine = math.cos(left_angle) + math.cos(right_angle)
    if math.hypot(sine, cosine) <= 1e-6:
        raise ValueError("schema-4 paired-eye alignment axes are inconsistent")
    rotation = math.atan2(sine, cosine)
    disagreement = abs(math.atan2(
        math.sin(left_angle - right_angle), math.cos(left_angle - right_angle)
    ))
    crop_side = SCHEMA_FOUR_CROP_SCALE * max(left_length, right_length)
    actual_rotation = finite_float(row, "alignment_rotation_deg")
    actual_disagreement = finite_float(row, "alignment_disagreement_deg")
    actual_crop_side = finite_float(row, "crop_side_px")
    if not -180 <= actual_rotation <= 180 \
            or not 0 <= actual_disagreement <= 180 \
            or actual_crop_side <= 0:
        raise ValueError("schema-4 alignment evidence is outside its valid range")
    expected_rotation = math.degrees(rotation)
    expected_disagreement = math.degrees(disagreement)
    source_diagonal = math.hypot(source_width, source_height)
    angle_tolerance = max(
        SCHEMA_FOUR_DERIVED_ANGLE_TOLERANCE_DEGREES,
        math.degrees(
            4 * source_diagonal * SCHEMA_FOUR_SERIALIZATION_RESOLUTION
            / min(left_length, right_length)
        ),
    )
    crop_side_tolerance = (
        4 * SCHEMA_FOUR_CROP_SCALE * source_diagonal
        * SCHEMA_FOUR_SERIALIZATION_RESOLUTION
        + 1e-9
    )
    values = (
        (actual_rotation, expected_rotation, angle_tolerance),
        (
            actual_disagreement,
            expected_disagreement,
            angle_tolerance,
        ),
        (actual_crop_side, crop_side, crop_side_tolerance),
        (
            left.crop_clipped_fraction,
            clipped_crop_fraction(
                left_center, rotation, crop_side, source_width, source_height
            ),
            SCHEMA_FOUR_DERIVED_CLIPPING_TOLERANCE,
        ),
        (
            right.crop_clipped_fraction,
            clipped_crop_fraction(
                right_center, rotation, crop_side, source_width, source_height
            ),
            SCHEMA_FOUR_DERIVED_CLIPPING_TOLERANCE,
        ),
    )
    if not all(
        math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)
        for actual, expected, tolerance in values
    ):
        raise ValueError("schema-4 derived alignment evidence does not match its source geometry")
    return CanonicalAlignmentEvidence(
        source_image_size=(source_width, source_height),
        left=left,
        right=right,
        rotation_degrees=actual_rotation,
        disagreement_degrees=actual_disagreement,
        crop_side_pixels=actual_crop_side,
    )


def parsed_utc_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"session {name} timestamp is invalid")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"session {name} timestamp is invalid") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"session {name} timestamp is invalid")
    return timestamp.astimezone(timezone.utc)


def validated_session_timestamps(metadata: dict[str, object]) -> tuple[str, str]:
    created = parsed_utc_timestamp(metadata.get("createdAt"), "creation")
    completed = parsed_utc_timestamp(metadata.get("completedAt"), "completion")
    if completed < created:
        raise ValueError("session completion precedes its creation")
    return created.isoformat(), completed.isoformat()


def validated_session_setup(
    metadata: dict[str, object], schema_version: int | None = None,
) -> tuple[dict[str, float], str, str, str, tuple[int, int] | None]:
    if schema_version is None:
        value = metadata.get("schemaVersion")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("session schema version is invalid")
        schema_version = value
    geometry = metadata.get("displayGeometry")
    if not isinstance(geometry, dict):
        raise ValueError("session display geometry is missing")
    names = ("displayWidthMM", "displayHeightMM", "viewingDistanceMM")
    values: dict[str, float] = {}
    for name in names:
        value = geometry.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise ValueError("session display geometry is invalid")
        values[name] = float(value)
    if values["displayWidthMM"] <= 1 or values["displayHeightMM"] <= 1 \
            or not 150 <= values["viewingDistanceMM"] <= 2000:
        raise ValueError("session display geometry is invalid")
    camera_format = metadata.get("cameraFormat")
    if not isinstance(camera_format, str) or not camera_format:
        raise ValueError("session camera format is invalid")
    if metadata.get("eyeImageWidth") != IMAGE_SIZE \
            or metadata.get("eyeImageHeight") != IMAGE_SIZE:
        raise ValueError("session eye image dimensions are invalid")
    setup = {
        "camera_format": camera_format,
        "display_geometry": values,
        "eye_image_height": IMAGE_SIZE,
        "eye_image_width": IMAGE_SIZE,
    }
    input_contract = LEGACY_INPUT_CONTRACT
    source_image_size: tuple[int, int] | None = None
    if schema_version == 4:
        source_width = validated_positive_integer(
            metadata.get("sourceImageWidth"), "source image width"
        )
        source_height = validated_positive_integer(
            metadata.get("sourceImageHeight"), "source image height"
        )
        if not matches_canonical_contract(
            metadata.get("cropContract"), CANONICAL_CROP_CONTRACT
        ):
            raise ValueError("session schema-4 crop contract is invalid")
        if not matches_canonical_contract(
            metadata.get("labelContract"), CANONICAL_LABEL_CONTRACT
        ):
            raise ValueError("session schema-4 label contract is invalid")
        if not matches_canonical_contract(
            metadata.get("headPoseContract"), CANONICAL_HEAD_POSE_CONTRACT
        ):
            raise ValueError("session schema-4 head-pose contract is invalid")
        source_image_size = (source_width, source_height)
        input_contract = CANONICAL_INPUT_CONTRACT
        setup.update({
            "source_image_width": source_width,
            "source_image_height": source_height,
            "crop_contract": CANONICAL_CROP_CONTRACT,
            "label_contract": CANONICAL_LABEL_CONTRACT,
            "head_pose_contract": CANONICAL_HEAD_POSE_CONTRACT,
        })
    binding = hashlib.sha256(
        json.dumps(setup, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        values,
        binding,
        input_contract,
        label_contract_digest(),
        source_image_size,
    )


def expected_target_label(
    split: str, target_id: int, geometry: dict[str, float]
) -> tuple[str, float, float, float, float, str]:
    if split not in {"training", "validation"}:
        raise ValueError("session split is invalid")
    if not 0 <= target_id < len(POSE_PROMPTS) * TARGETS_PER_POSE:
        raise ValueError("target id is invalid")
    pose_index, position = divmod(target_id, TARGETS_PER_POSE)
    pose_prompt = POSE_PROMPTS[pose_index]
    if position in {0, TARGETS_PER_POSE - 1}:
        return "lens", 0.5, 0.0, 0.0, 0.0, pose_prompt
    order = list(TRAINING_GRID_ORDER)
    if split == "validation":
        order.reverse()
    shift = pose_index * 4
    order = order[shift:] + order[:shift]
    grid_index = order[position - 1]
    row, column = divmod(grid_index, len(GRID_FRACTIONS))
    target_x = GRID_FRACTIONS[column]
    target_y = GRID_FRACTIONS[row]
    right = (target_x - 0.5) * geometry["displayWidthMM"]
    up = -target_y * geometry["displayHeightMM"]
    distance = geometry["viewingDistanceMM"]
    target_yaw = math.degrees(math.atan2(right, distance))
    target_pitch = math.degrees(math.atan2(up, distance))
    return "screen", target_x, target_y, target_yaw, target_pitch, pose_prompt


def finished_session_metadata(root: Path) -> list[tuple[Path, dict[str, object]]]:
    metadata_files = sorted(root.rglob("session.json"))
    if not metadata_files:
        raise ValueError(f"no session.json files found under {root}")
    result: list[tuple[Path, dict[str, object]]] = []
    seen_session_ids: set[str] = set()
    for metadata_path in metadata_files:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "finished":
            continue
        session_id = str(metadata["sessionID"])
        if session_id in seen_session_ids:
            raise ValueError(f"duplicate finished session id: {session_id}")
        seen_session_ids.add(session_id)
        result.append((metadata_path, metadata))
    return result


def finished_session_ids(root: Path) -> list[str]:
    return [
        str(metadata["sessionID"])
        for _, metadata in finished_session_metadata(root)
    ]


def load_samples(
    root: Path,
    split: str,
    selected_session_ids: set[str] | None = None,
    minimum_face_confidence: float = MINIMUM_FACE_CONFIDENCE,
) -> tuple[list[Sample], set[str], list[str]]:
    samples: list[Sample] = []
    participants: set[str] = set()
    loaded_session_ids: list[str] = []

    for metadata_path, metadata in finished_session_metadata(root):
        session_id = str(metadata["sessionID"])
        stored_split = str(metadata["split"])
        schema_version = int(metadata["schemaVersion"])
        if schema_version not in {1, 2, 3, 4}:
            raise ValueError(f"unsupported gaze dataset schema: {schema_version}")
        (
            geometry,
            setup_binding,
            input_contract,
            session_label_contract_sha256,
            source_image_size,
        ) = validated_session_setup(metadata, schema_version)
        session_created_at, _ = validated_session_timestamps(metadata)
        if selected_session_ids is None and stored_split != split:
            continue
        if selected_session_ids is not None and session_id not in selected_session_ids:
            continue
        participant_id = str(metadata["participantID"])
        participant_binding = hashlib.sha256(participant_id.encode()).hexdigest()
        manifest_path = metadata_path.with_name("manifest.csv")
        with manifest_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = REQUIRED_COLUMNS | (
                SCHEMA_FOUR_COLUMNS if schema_version == 4 else set()
            )
            fieldnames = reader.fieldnames or []
            missing = required_columns.difference(fieldnames)
            if missing:
                raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
            if schema_version == 4 and (
                len(fieldnames) != len(required_columns)
                or set(fieldnames) != required_columns
            ):
                raise ValueError("schema-4 manifest columns do not match the frozen contract")
            rows = list(reader)
            if schema_version == 4 and any(
                None in row or any(value is None for value in row.values())
                for row in rows
            ):
                raise ValueError("schema-4 manifest rows do not match the frozen contract")
        session_metadata_sha256 = file_digest(metadata_path)
        manifest_sha256 = file_digest(manifest_path)
        expected = int(metadata["capturedSamples"])
        samples_per_target = int(metadata["samplesPerTarget"])
        target_count = int(metadata["targetCount"])
        if samples_per_target != 6 or target_count != len(POSE_PROMPTS) * TARGETS_PER_POSE:
            raise ValueError(f"unsupported collection plan in {session_id}")
        if expected != samples_per_target * target_count:
            raise ValueError(f"{session_id} is not a complete collection")
        if expected != len(rows):
            raise ValueError(
                f"{session_id} metadata records {expected} samples but manifest has {len(rows)}"
            )
        sample_numbers = [int(row["sample"]) for row in rows]
        if sample_numbers != list(range(1, expected + 1)):
            raise ValueError(f"{session_id} sample sequence is incomplete")
        target_counts: dict[int, int] = {}
        previous_frame_id = -1
        previous_elapsed = -math.inf
        for row in rows:
            if row["participant_id"] != participant_id \
                    or row["session_id"] != session_id or row["split"] != stored_split:
                raise ValueError(f"manifest identity mismatch in {manifest_path}")
            if int(row["schema_version"]) != schema_version:
                raise ValueError(f"manifest schema mismatch in {manifest_path}")
            left_path = metadata_path.parent / row["left_image"]
            right_path = metadata_path.parent / row["right_image"]
            if not left_path.is_file() or not right_path.is_file():
                raise ValueError(f"missing eye image for sample {row['sample']} in {session_id}")
            target_id = int(row["target_id"])
            if not 0 <= target_id < target_count:
                raise ValueError(f"invalid target id in {manifest_path}")
            if target_id != (int(row["sample"]) - 1) // samples_per_target:
                raise ValueError(f"manifest target order mismatch in {manifest_path}")
            target_counts[target_id] = target_counts.get(target_id, 0) + 1
            face_confidence = finite_float(row, "face_conf")
            left_openness = finite_float(row, "open_l")
            right_openness = finite_float(row, "open_r")
            if schema_version == 4 and (
                not 0 <= face_confidence <= 1
                or not 0 <= left_openness <= 1
                or not 0 <= right_openness <= 1
            ):
                raise ValueError("schema-4 tracking quality is outside its valid range")
            if schema_version == 4:
                frame_id = finite_integer(row, "frame_id")
                elapsed = finite_float(row, "elapsed_s")
                if elapsed < 0:
                    raise ValueError("schema-4 elapsed time is invalid")
                if frame_id <= previous_frame_id or elapsed <= previous_elapsed:
                    raise ValueError("schema-4 frame timing is not strictly increasing")
                previous_frame_id = frame_id
                previous_elapsed = elapsed
            head_yaw = finite_float(row, "head_yaw_deg")
            head_pitch = finite_float(row, "head_pitch_deg")
            head_roll = finite_float(row, "head_roll_deg")
            target_yaw = finite_float(row, "target_yaw_deg")
            target_pitch = finite_float(row, "target_pitch_deg")
            target_x = finite_float(row, "target_x")
            target_y = finite_float(row, "target_y")
            expected = expected_target_label(stored_split, target_id, geometry)
            expected_kind, expected_x, expected_y, expected_yaw, expected_pitch, expected_pose = (
                expected
            )
            label_values = (
                (target_x, expected_x),
                (target_y, expected_y),
                (target_yaw, expected_yaw),
                (target_pitch, expected_pitch),
            )
            labels_match = all(
                math.isclose(
                    actual, planned, rel_tol=0,
                    abs_tol=TARGET_LABEL_ABSOLUTE_TOLERANCE,
                )
                for actual, planned in label_values
            )
            if row["target_kind"] != expected_kind \
                    or row["pose_prompt"] != expected_pose \
                    or not labels_match:
                raise ValueError(f"manifest target plan mismatch in {manifest_path}")
            if schema_version == 1 and expected_kind == "lens" \
                    and target_id % TARGETS_PER_POSE == 0:
                continue
            canonical_alignment = None
            if schema_version == 4:
                if source_image_size is None:
                    raise ValueError("session schema-4 source image size is missing")
                canonical_alignment = schema_four_alignment_evidence(
                    row, *source_image_size
                )
            if face_confidence < minimum_face_confidence \
                    or left_openness < MINIMUM_OPENNESS \
                    or right_openness < MINIMUM_OPENNESS \
                    or abs(head_roll) > MAXIMUM_ABSOLUTE_HEAD_ROLL_DEGREES:
                continue
            samples.append(
                Sample(
                    session_id=session_id,
                    split=split,
                    target_id=target_id,
                    target_kind=row["target_kind"],
                    left_path=left_path,
                    right_path=right_path,
                    head_pose=(head_yaw, head_pitch, head_roll),
                    gaze_degrees=(target_yaw, target_pitch),
                    pose_prompt=row["pose_prompt"],
                    target_position=(target_x, target_y),
                    face_confidence=face_confidence,
                    minimum_openness=min(left_openness, right_openness),
                    schema_version=schema_version,
                    participant_binding=participant_binding,
                    setup_binding=setup_binding,
                    session_created_at=session_created_at,
                    session_metadata_sha256=session_metadata_sha256,
                    manifest_sha256=manifest_sha256,
                    input_contract=input_contract,
                    label_contract_sha256=session_label_contract_sha256,
                    canonical_alignment=canonical_alignment,
                )
            )
        if target_counts != {target_id: samples_per_target for target_id in range(target_count)}:
            raise ValueError(f"{session_id} target sequence is incomplete")
        participants.add(participant_id)
        loaded_session_ids.append(session_id)
    if not samples:
        raise ValueError(f"no finished {split} sessions found under {root}")
    if selected_session_ids is not None \
            and selected_session_ids != set(loaded_session_ids):
        raise ValueError(f"not every requested {split} session was found")
    return samples, participants, loaded_session_ids


def validate_pose_coverage(samples: list[Sample], expected_sessions: Iterable[str]) -> None:
    expected_sessions = list(expected_sessions)
    counts: dict[tuple[str, int], int] = {}
    lens_counts: dict[tuple[str, int], int] = {}
    lens_targets: dict[tuple[str, int], set[int]] = {}
    for sample in samples:
        pose_index = sample.target_id // TARGETS_PER_POSE
        key = (sample.session_id, pose_index)
        counts[key] = counts.get(key, 0) + 1
        if sample.target_kind == "lens":
            lens_counts[key] = lens_counts.get(key, 0) + 1
            lens_targets.setdefault(key, set()).add(sample.target_id % TARGETS_PER_POSE)
    incomplete = {
        session_id: [
            pose_index for pose_index in range(5)
            if counts.get((session_id, pose_index), 0) < MINIMUM_VALIDATION_SAMPLES_PER_POSE
        ]
        for session_id in expected_sessions
    }
    incomplete = {session: poses for session, poses in incomplete.items() if poses}
    if incomplete:
        raise ValueError(f"validation pose coverage is incomplete after filtering: {incomplete}")
    insufficient_lens = {
        session_id: [
            pose_index for pose_index in range(len(POSE_PROMPTS))
            if lens_counts.get((session_id, pose_index), 0)
            < MINIMUM_VALIDATION_LENS_SAMPLES_PER_POSE
            or len(lens_targets.get((session_id, pose_index), set()))
            < MINIMUM_VALIDATION_LENS_TARGETS_PER_POSE
        ]
        for session_id in expected_sessions
    }
    insufficient_lens = {
        session: poses for session, poses in insufficient_lens.items() if poses
    }
    lens_totals = {
        session_id: sum(
            lens_counts.get((session_id, pose_index), 0)
            for pose_index in range(len(POSE_PROMPTS))
        )
        for session_id in expected_sessions
    }
    insufficient_totals = {
        session_id: count for session_id, count in lens_totals.items()
        if count < MINIMUM_VALIDATION_LENS_SAMPLES
    }
    if insufficient_lens or insufficient_totals:
        raise ValueError(
            "validation lens coverage is incomplete after filtering: "
            f"poses={insufficient_lens}, totals={insufficient_totals}"
        )
    for session_id in expected_sessions:
        pose_samples = [
            [
                sample for sample in samples
                if sample.session_id == session_id
                and sample.target_id // TARGETS_PER_POSE == pose_index
            ]
            for pose_index in range(len(POSE_PROMPTS))
        ]
        neutral_yaw = statistics.fmean(sample.head_pose[0] for sample in pose_samples[0])
        neutral_pitch = statistics.fmean(sample.head_pose[1] for sample in pose_samples[0])
        left_change = statistics.fmean(
            sample.head_pose[0] for sample in pose_samples[1]
        ) - neutral_yaw
        right_change = statistics.fmean(
            sample.head_pose[0] for sample in pose_samples[2]
        ) - neutral_yaw
        up_change = statistics.fmean(
            sample.head_pose[1] for sample in pose_samples[3]
        ) - neutral_pitch
        down_change = statistics.fmean(
            sample.head_pose[1] for sample in pose_samples[4]
        ) - neutral_pitch
        if abs(left_change) < MINIMUM_VALIDATION_HORIZONTAL_POSE_SEPARATION_DEGREES \
                or abs(right_change) \
                < MINIMUM_VALIDATION_HORIZONTAL_POSE_SEPARATION_DEGREES \
                or left_change * right_change >= 0 \
                or up_change > -MINIMUM_VALIDATION_VERTICAL_POSE_SEPARATION_DEGREES \
                or down_change < MINIMUM_VALIDATION_VERTICAL_POSE_SEPARATION_DEGREES:
            raise ValueError(
                "validation head-pose coverage does not separate the prompted poses"
            )


def validate_numeric_pitch_contract(
    samples: list[Sample], expected_sessions: Iterable[str],
    require_head_pitch: bool = True,
) -> None:
    expected_sessions = list(expected_sessions)
    eligible_head_pose_sessions = 0
    for session_id in expected_sessions:
        session_samples = [sample for sample in samples if sample.session_id == session_id]
        if not session_samples:
            raise ValueError("numeric pitch verification is missing a selected session")
        lens_labels = [
            sample.gaze_degrees
            for sample in session_samples if sample.target_kind == "lens"
        ]
        screen_samples = [
            sample for sample in session_samples if sample.target_kind == "screen"
        ]
        if not lens_labels or not screen_samples \
                or not all(
                    math.isclose(value, 0, rel_tol=0,
                                 abs_tol=TARGET_LABEL_ABSOLUTE_TOLERANCE)
                    for label in lens_labels for value in label
                ) \
                or not all(sample.gaze_degrees[1] < 0 for sample in screen_samples):
            raise ValueError("numeric lens and screen label contract is invalid")
        yaw_by_x: dict[float, list[float]] = {}
        pitch_by_y: dict[float, list[float]] = {}
        for sample in screen_samples:
            yaw_by_x.setdefault(sample.target_position[0], []).append(
                sample.gaze_degrees[0]
            )
            pitch_by_y.setdefault(sample.target_position[1], []).append(
                sample.gaze_degrees[1]
            )
        horizontal = [
            statistics.fmean(yaw_by_x[position])
            for position in sorted(yaw_by_x)
        ]
        vertical = [
            statistics.fmean(pitch_by_y[position])
            for position in sorted(pitch_by_y)
        ]
        if len(horizontal) < 2 \
                or not all(left < right for left, right in zip(
                    horizontal, horizontal[1:]
                )):
            raise ValueError("numeric target yaw is not monotonic across the display")
        if len(vertical) < 2 \
                or not all(upper > lower for upper, lower in zip(
                    vertical, vertical[1:]
                )):
            raise ValueError("numeric target pitch is not monotonic down the display")

        schema_versions = {sample.schema_version for sample in session_samples}
        if max(schema_versions, default=0) < 2:
            continue
        eligible_head_pose_sessions += 1
        pose_pitches = {
            pose_index: [
                sample.head_pose[1] for sample in session_samples
                if sample.target_id // TARGETS_PER_POSE == pose_index
            ]
            for pose_index in (0, 3, 4)
        }
        if any(not values for values in pose_pitches.values()):
            raise ValueError("numeric head-pitch verification lacks a pose block")
        neutral = statistics.median(pose_pitches[0])
        up_change = statistics.median(pose_pitches[3]) - neutral
        down_change = statistics.median(pose_pitches[4]) - neutral
        if up_change > -MINIMUM_VALIDATION_VERTICAL_POSE_SEPARATION_DEGREES \
                or down_change < MINIMUM_VALIDATION_VERTICAL_POSE_SEPARATION_DEGREES:
            raise ValueError(
                "numeric head-pitch orientation must be negative for up and positive for down"
            )
    if require_head_pitch and eligible_head_pose_sessions != len(expected_sessions):
        raise ValueError("numeric head-pitch verification requires schema-2-or-newer sessions")


def gaze_vectors(angles: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(angles)
    yaw = radians[:, 0]
    pitch = radians[:, 1]
    return np.stack(
        (np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)),
        axis=1,
    )


def native_degree_vectors(angles: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(angles)
    yaw = radians[:, 0]
    pitch = radians[:, 1]
    return torch.stack(
        (
            torch.sin(yaw) * torch.cos(pitch),
            torch.sin(pitch),
            torch.cos(yaw) * torch.cos(pitch),
        ),
        dim=1,
    )


def angular_cosine_loss(
    predicted_degrees: torch.Tensor, expected_degrees: torch.Tensor,
) -> torch.Tensor:
    predicted = native_degree_vectors(predicted_degrees)
    expected = native_degree_vectors(expected_degrees)
    return (1 - torch.sum(predicted * expected, dim=1)).mean()


def openvino_target_vectors(labels: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(labels)
    yaw = radians[:, 0]
    pitch = radians[:, 1]
    native = torch.stack(
        (
            torch.sin(yaw) * torch.cos(pitch),
            torch.sin(pitch),
            torch.cos(yaw) * torch.cos(pitch),
        ),
        dim=1,
    )
    return native * native.new_tensor([-1, 1, -1])


def openvino_vectors_to_degrees(vectors: torch.Tensor) -> torch.Tensor:
    native = functional.normalize(vectors.reshape(-1, 3), dim=1)
    native = native * native.new_tensor([-1, 1, -1])
    yaw = torch.rad2deg(torch.atan2(native[:, 0], native[:, 2]))
    pitch = torch.rad2deg(torch.asin(native[:, 1].clamp(-1, 1)))
    return torch.stack((yaw, pitch), dim=1)


def aggregate_angular_training_loss(
    per_sample_losses: torch.Tensor, formulation: str
) -> torch.Tensor:
    mean_loss = per_sample_losses.mean()
    if formulation == "mean-angular":
        return mean_loss
    if formulation == "tail-angular":
        tail_count = max(1, math.ceil(per_sample_losses.numel() * TAIL_LOSS_FRACTION))
        tail_loss = torch.topk(per_sample_losses, tail_count).values.mean()
        return (1 - TAIL_LOSS_WEIGHT) * mean_loss + TAIL_LOSS_WEIGHT * tail_loss
    raise ValueError(f"unsupported pretrained loss formulation: {formulation}")


def training_loss_configuration(
    pretrained: bool, pretrained_formulation: str,
) -> dict[str, float | str | None]:
    if pretrained_formulation not in {"mean-angular", "tail-angular"}:
        raise ValueError("unsupported pretrained loss formulation")
    if not pretrained and pretrained_formulation != "mean-angular":
        raise ValueError("pretrained loss formulation requires a pretrained model")
    tail_weighted = pretrained and pretrained_formulation == "tail-angular"
    return {
        "formulation": "tail-angular-cosine" if tail_weighted else "angular-cosine",
        "tail_fraction": TAIL_LOSS_FRACTION if tail_weighted else None,
        "tail_weight": TAIL_LOSS_WEIGHT if tail_weighted else None,
    }


def angular_errors(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    products = np.sum(gaze_vectors(predicted) * gaze_vectors(expected), axis=1)
    return np.rad2deg(np.arccos(np.clip(products, -1.0, 1.0)))


def percentile(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(values, fraction * 100, method="linear"))


def evaluation_from_predictions(
    predicted_array: np.ndarray,
    expected_array: np.ndarray,
    lens_mask: np.ndarray,
) -> Evaluation:
    errors = angular_errors(predicted_array, expected_array)
    lens_errors = errors[lens_mask]
    absolute = np.abs(predicted_array - expected_array)
    return Evaluation(
        count=len(errors),
        angular_median_degrees=float(np.median(errors)),
        angular_p95_degrees=percentile(errors, 0.95),
        lens_count=len(lens_errors),
        lens_p95_degrees=percentile(lens_errors, 0.95),
        yaw_mae_degrees=float(np.mean(absolute[:, 0])),
        pitch_mae_degrees=float(np.mean(absolute[:, 1])),
    )


def predict_degrees(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pretrained: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    lens_masks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for left, right, pose, labels, is_lens in loader:
            if pretrained:
                result = openvino_vectors_to_degrees(model(
                    rgb_to_bgr(left.to(device)) * 255,
                    rgb_to_bgr(right.to(device)) * 255,
                    pose.to(device),
                ))
            else:
                result = model(left.to(device), right.to(device), pose.to(device))
            predictions.append(result.cpu().numpy())
            expected.append(labels.numpy())
            lens_masks.append(is_lens.numpy().astype(bool))
    return (
        np.concatenate(predictions),
        np.concatenate(expected),
        np.concatenate(lens_masks),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Evaluation:
    return evaluation_from_predictions(*predict_degrees(model, loader, device, False))


def evaluate_pretrained(model: nn.Module, loader: DataLoader,
                        device: torch.device) -> Evaluation:
    return evaluation_from_predictions(*predict_degrees(model, loader, device, True))


def evaluate_samples(
    model: nn.Module,
    samples: list[Sample],
    batch_size: int,
    device: torch.device,
    pretrained: bool,
) -> Evaluation:
    loader = DataLoader(
        GazeDataset(samples, augment=False), batch_size=batch_size, shuffle=False
    )
    return evaluation_from_predictions(*predict_degrees(model, loader, device, pretrained))


def evaluate_sessions(
    model: nn.Module,
    samples: list[Sample],
    session_ids: Iterable[str],
    batch_size: int,
    device: torch.device,
    pretrained: bool,
) -> dict[str, Evaluation]:
    grouped = {
        session_id: [sample for sample in samples if sample.session_id == session_id]
        for session_id in session_ids
    }
    missing = [session_id for session_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"development sessions have no retained samples: {missing}")
    return {
        session_id: evaluate_samples(model, grouped[session_id], batch_size, device, pretrained)
        for session_id in sorted(grouped)
    }


def subset_metrics(
    predicted: np.ndarray,
    expected: np.ndarray,
    lens_mask: np.ndarray,
    indices: list[int],
) -> dict[str, float | int | None]:
    subset_predicted = predicted[indices]
    subset_expected = expected[indices]
    subset_lens_mask = lens_mask[indices]
    errors = angular_errors(subset_predicted, subset_expected)
    lens_errors = errors[subset_lens_mask]
    absolute = np.abs(subset_predicted - subset_expected)
    return {
        "count": len(indices),
        "angular_median_degrees": float(np.median(errors)),
        "angular_p95_degrees": percentile(errors, 0.95),
        "lens_count": len(lens_errors),
        "lens_p95_degrees": (
            percentile(lens_errors, 0.95) if len(lens_errors) else None
        ),
        "yaw_mae_degrees": float(np.mean(absolute[:, 0])),
        "pitch_mae_degrees": float(np.mean(absolute[:, 1])),
    }


def bucket(value: float, boundaries: tuple[float, ...]) -> str:
    for lower, upper in zip(boundaries, boundaries[1:]):
        if lower <= value < upper:
            lower_label = "-inf" if math.isinf(lower) else f"{lower:g}"
            upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"[{lower_label},{upper_label})"
    raise ValueError(f"value outside bucket boundaries: {value}")


def grouped_metrics(
    samples: list[Sample],
    predicted: np.ndarray,
    expected: np.ndarray,
    lens_mask: np.ndarray,
    key: Callable[[Sample], object],
) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        label = str(key(sample))
        groups.setdefault(label, []).append(index)
    return {
        label: subset_metrics(predicted, expected, lens_mask, indices)
        for label, indices in sorted(groups.items())
    }


def numeric_evidence_summary(values: Iterable[float]) -> dict[str, float | int]:
    values = list(values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("schema-4 diagnostic evidence is invalid")
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def schema_four_quality_diagnostics(samples: list[Sample]) -> dict[str, object] | None:
    if not samples or any(sample.schema_version != 4 for sample in samples):
        return None
    alignments = [sample.canonical_alignment for sample in samples]
    if any(alignment is None for alignment in alignments):
        raise ValueError("schema-4 diagnostic evidence is missing")
    values = [alignment for alignment in alignments if alignment is not None]
    pupil_source_pairs: dict[str, int] = {}
    for alignment in values:
        key = f"{alignment.left.pupil_source}+{alignment.right.pupil_source}"
        pupil_source_pairs[key] = pupil_source_pairs.get(key, 0) + 1
    return {
        "crop_clipped_fraction": numeric_evidence_summary(
            fraction
            for alignment in values
            for fraction in (
                alignment.left.crop_clipped_fraction,
                alignment.right.crop_clipped_fraction,
            )
        ),
        "alignment_disagreement_degrees": numeric_evidence_summary(
            alignment.disagreement_degrees for alignment in values
        ),
        "crop_side_pixels": numeric_evidence_summary(
            alignment.crop_side_pixels for alignment in values
        ),
        "minimum_contour_point_count": numeric_evidence_summary(
            float(min(
                alignment.left.contour_point_count,
                alignment.right.contour_point_count,
            ))
            for alignment in values
        ),
        "pupil_source_pair_counts": dict(sorted(pupil_source_pairs.items())),
    }


def diagnostic_breakdowns(
    model: nn.Module,
    samples: list[Sample],
    session_ids: Iterable[str],
    batch_size: int,
    device: torch.device,
    pretrained: bool,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    angle_boundaries = (-math.inf, -15.0, -5.0, 5.0, 15.0, math.inf)
    confidence_boundaries = (0.70, 0.75, 0.80, 0.85, 0.90, math.inf)
    openness_boundaries = (0.40, 0.60, 0.80, 1.01, math.inf)
    for session_id in sorted(session_ids):
        session_samples = [sample for sample in samples if sample.session_id == session_id]
        loader = DataLoader(
            GazeDataset(session_samples, augment=False),
            batch_size=batch_size,
            shuffle=False,
        )
        predicted, expected, lens_mask = predict_degrees(
            model, loader, device, pretrained
        )
        schema_four_quality = schema_four_quality_diagnostics(session_samples)
        result[session_id] = {
            "pose_block": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: f"{sample.target_id // TARGETS_PER_POSE}:{sample.pose_prompt}",
            ),
            "target_location": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: (
                    f"{sample.target_id % TARGETS_PER_POSE:02d}:{sample.target_kind}:"
                    f"{sample.target_position[0]:.2f},{sample.target_position[1]:.2f}"
                ),
            ),
            "target_kind": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: sample.target_kind,
            ),
            "target_yaw_degrees": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: f"{sample.gaze_degrees[0]:.6f}",
            ),
            "target_pitch_degrees": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: f"{sample.gaze_degrees[1]:.6f}",
            ),
            "head_yaw_bucket_degrees": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: bucket(sample.head_pose[0], angle_boundaries),
            ),
            "head_pitch_bucket_degrees": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: bucket(sample.head_pose[1], angle_boundaries),
            ),
            "head_roll_bucket_degrees": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: bucket(sample.head_pose[2], angle_boundaries),
            ),
            "face_confidence_bucket": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: bucket(sample.face_confidence, confidence_boundaries),
            ),
            "minimum_eye_openness_bucket": grouped_metrics(
                session_samples, predicted, expected, lens_mask,
                lambda sample: bucket(sample.minimum_openness, openness_boundaries),
            ),
            "schema_four_quality": schema_four_quality,
            "unavailable_indicators": (
                ["occlusion_score"]
                if schema_four_quality is not None
                else ["crop_clipping", "alignment_quality", "occlusion"]
            ),
        }
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def quality_gates() -> QualityGates:
    return QualityGates(
        maximum_median_degrees=CANONICAL_MAXIMUM_MEDIAN_DEGREES,
        maximum_p95_degrees=CANONICAL_MAXIMUM_P95_DEGREES,
        maximum_lens_p95_degrees=CANONICAL_MAXIMUM_LENS_P95_DEGREES,
    )


def selection_components(
    session_evaluations: dict[str, Evaluation], gates: QualityGates
) -> tuple[float, float, float, float]:
    if not session_evaluations:
        raise ValueError("checkpoint selection requires development sessions")
    for evaluation in session_evaluations.values():
        values = (
            evaluation.angular_median_degrees,
            evaluation.angular_p95_degrees,
            evaluation.lens_p95_degrees,
            evaluation.yaw_mae_degrees,
            evaluation.pitch_mae_degrees,
        )
        if evaluation.count < 1 or evaluation.lens_count < 1 \
                or not all(math.isfinite(value) for value in values):
            raise ValueError("checkpoint selection metrics must be finite and nonempty")
    worst_median_ratio = max(
        evaluation.angular_median_degrees / gates.maximum_median_degrees
        for evaluation in session_evaluations.values()
    )
    worst_p95_ratio = max(
        evaluation.angular_p95_degrees / gates.maximum_p95_degrees
        for evaluation in session_evaluations.values()
    )
    worst_lens_ratio = max(
        evaluation.lens_p95_degrees / gates.maximum_lens_p95_degrees
        for evaluation in session_evaluations.values()
    )
    return (
        max(worst_median_ratio, worst_p95_ratio, worst_lens_ratio),
        worst_lens_ratio,
        worst_p95_ratio,
        worst_median_ratio,
    )


def epoch_evaluation(
    epoch: int,
    learning_rate: float,
    session_evaluations: dict[str, Evaluation],
    gates: QualityGates,
) -> EpochEvaluation:
    score, lens, p95, median = selection_components(session_evaluations, gates)
    return EpochEvaluation(
        epoch=epoch,
        learning_rate=learning_rate,
        session_evaluations=session_evaluations,
        selection_score=score,
        worst_lens_ratio=lens,
        worst_p95_ratio=p95,
        worst_median_ratio=median,
    )


def select_epoch(evaluated_epochs: list[EpochEvaluation]) -> EpochEvaluation:
    if not evaluated_epochs:
        raise ValueError("no development epochs were evaluated")
    return min(
        evaluated_epochs,
        key=lambda result: (
            result.selection_score,
            result.worst_lens_ratio,
            result.worst_p95_ratio,
            result.worst_median_ratio,
            result.epoch,
        ),
    )


def snapshot_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def restore_model_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def should_evaluate_epoch(epoch: int, epochs: int, interval: int) -> bool:
    return epoch == 1 or epoch == epochs or epoch % interval == 0


def learning_rate_scheduler(
    optimiser: torch.optim.Optimizer,
    schedule: str,
    epochs: int,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if schedule == "constant":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=minimum_learning_rate
        )
    raise ValueError(f"unsupported learning-rate schedule: {schedule}")


def require_untouched_final_session(
    validation_sessions: Iterable[str],
    training_sessions: Iterable[str],
    development_sessions: Iterable[str],
) -> None:
    consumed = set(training_sessions) | set(development_sessions)
    if consumed & set(validation_sessions):
        raise ValueError("final validation session was already consumed by the checkpoint")


def require_final_evaluation_output(
    checkpoint_path: Path, output_path: Path
) -> None:
    required_output = checkpoint_path.parent / FINAL_EVALUATION_FILENAME
    if output_path.resolve() != required_output.resolve():
        raise ValueError(f"final evaluation output must be {required_output}")


def validated_completed_session_ledger(value: object) -> list[str]:
    if not isinstance(value, list) \
            or not value \
            or not all(isinstance(session_id, str) for session_id in value) \
            or len(value) != len(set(value)):
        raise ValueError("checkpoint does not contain a valid completed-session ledger")
    return value


def require_stable_completed_session_ledger(
    initial_sessions: Iterable[str], frozen_sessions: Iterable[str],
    operation: str = "training",
) -> list[str]:
    initial = sorted(initial_sessions)
    frozen = sorted(frozen_sessions)
    if initial != frozen:
        raise ValueError(f"completed-session ledger changed during {operation}")
    return frozen


def require_single_new_completed_session(
    validation_sessions: Iterable[str],
    known_completed_sessions: Iterable[str],
    current_completed_sessions: Iterable[str],
) -> None:
    validation = set(validation_sessions)
    known = set(known_completed_sessions)
    current = set(current_completed_sessions)
    if not known.issubset(current):
        raise ValueError("a session from the frozen completed-session ledger is missing")
    if current - known != validation:
        raise ValueError(
            "final evaluation requires exactly the one completed session recorded after freezing"
        )


def require_single_new_completed_session_hashes(
    validation_sessions: Iterable[str],
    known_completed_session_sha256: Iterable[str],
    current_completed_sessions: Iterable[str],
) -> None:
    validation = {session_id_digest(value) for value in validation_sessions}
    known = {
        validated_sha256(value, "completed session")
        for value in known_completed_session_sha256
    }
    current = {session_id_digest(value) for value in current_completed_sessions}
    if not known.issubset(current):
        raise ValueError("a session from the frozen completed-session snapshot is missing")
    if current - known != validation:
        raise ValueError(
            "final evaluation requires exactly the one completed session recorded after freezing"
        )


def require_final_session_created_after_freeze(
    validation: Iterable[Sample], frozen_at: object
) -> None:
    created_values = {sample.session_created_at for sample in validation}
    if len(created_values) != 1 or "" in created_values:
        raise ValueError("final session creation timestamp is invalid")
    created = parsed_utc_timestamp(next(iter(created_values)), "creation")
    frozen = parsed_utc_timestamp(frozen_at, "checkpoint freeze")
    if created <= frozen:
        raise ValueError("final session began before the candidate was frozen")


def final_evaluation_ledger_root() -> Path:
    return Path(__file__).resolve().parent / "runs"


def claim_final_evaluation_session(
    root: Path, session_id: str, candidate_identity_sha256: str,
    checkpoint_sha256: str,
) -> str:
    if not session_id:
        raise ValueError("final validation session id is invalid")
    candidate_identity_sha256 = validated_sha256(
        candidate_identity_sha256, "candidate identity"
    )
    validated_sha256(checkpoint_sha256, "checkpoint")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    lock_path = root / FINAL_EVALUATION_LEDGER_LOCK_FILENAME
    ledger_path = root / FINAL_EVALUATION_LEDGER_FILENAME
    session_sha256 = session_id_digest(session_id)
    with lock_path.open("a+") as lock:
        protect_private_artifact(lock_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
        else:
            ledger = {"schema_version": 2, "claims": []}
        if not isinstance(ledger, dict):
            raise ValueError("final-evaluation ledger is invalid")
        claims = ledger.get("claims")
        if ledger.get("schema_version") != 2 or not isinstance(claims, list) \
                or not all(isinstance(claim, dict) for claim in claims):
            raise ValueError("final-evaluation ledger is invalid")
        try:
            claimed_sessions = []
            claimed_candidates = []
            for claim in claims:
                if set(claim) != {
                    "session_sha256", "candidate_identity_sha256",
                    "checkpoint_sha256", "claimed_at",
                }:
                    raise ValueError
                claimed_sessions.append(
                    validated_sha256(claim["session_sha256"], "session")
                )
                claimed_candidates.append(
                    validated_sha256(
                        claim["candidate_identity_sha256"], "candidate identity"
                    )
                )
                validated_sha256(claim["checkpoint_sha256"], "checkpoint")
                parsed_utc_timestamp(claim["claimed_at"], "claim")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("final-evaluation ledger is invalid") from error
        if len(claimed_sessions) != len(set(claimed_sessions)) \
                or len(claimed_candidates) != len(set(claimed_candidates)):
            raise ValueError("final-evaluation ledger is invalid")
        if session_sha256 in claimed_sessions:
            raise ValueError("final validation session has already been evaluated")
        if candidate_identity_sha256 in claimed_candidates:
            raise ValueError("frozen candidate has already used a final validation session")
        claims.append({
            "session_sha256": session_sha256,
            "candidate_identity_sha256": candidate_identity_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "claimed_at": datetime.now(timezone.utc).isoformat(),
        })
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=root, prefix=".aspectus-final-ledger-",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.fchmod(temporary.fileno(), 0o600)
                json.dump(ledger, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                sync_to_permanent_storage(temporary.fileno())
            os.replace(temporary_path, ledger_path)
            protect_private_artifact(ledger_path)
            durably_sync_published_path(ledger_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return session_sha256


def train_model(
    training: list[Sample],
    validation: list[Sample],
    validation_sessions: list[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    learning_rate_schedule: str,
    minimum_learning_rate: float,
    augmentation_strength: str,
    weight_decay: float,
    evaluation_interval: int,
    seed: int,
    gates: QualityGates,
) -> TrainingOutcome:
    seed_everything(seed)
    device = execution_device()
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        GazeDataset(
            training, augment=True, augmentation_strength=augmentation_strength
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = GazeEstimator().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = learning_rate_scheduler(
        optimiser, learning_rate_schedule, epochs, minimum_learning_rate
    )
    evaluated_epochs: list[EpochEvaluation] = []
    selected_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for left, right, pose, labels, _ in training_loader:
            optimiser.zero_grad(set_to_none=True)
            output = model(left.to(device), right.to(device), pose.to(device))
            loss = angular_cosine_loss(output, labels.to(device))
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch {epoch:03d} loss={statistics.fmean(losses):.4f}")
        if should_evaluate_epoch(epoch, epochs, evaluation_interval):
            sessions = evaluate_sessions(
                model, validation, validation_sessions, batch_size, device, False
            )
            result = epoch_evaluation(
                epoch, optimiser.param_groups[0]["lr"], sessions, gates
            )
            evaluated_epochs.append(result)
            if select_epoch(evaluated_epochs).epoch == epoch:
                selected_state = snapshot_model_state(model)
            print(f"epoch {epoch:03d} development_score={result.selection_score:.6f}")
        if scheduler is not None:
            scheduler.step()

    selected = select_epoch(evaluated_epochs)
    if selected_state is None:
        raise RuntimeError("selected model state was not retained")
    restore_model_state(model, selected_state)
    selected_sessions = evaluate_sessions(
        model, validation, validation_sessions, batch_size, device, False
    )
    selected_evaluation = evaluate_samples(model, validation, batch_size, device, False)
    model.cpu().eval()
    return TrainingOutcome(
        model=model,
        selected_epoch=selected.epoch,
        selected_evaluation=selected_evaluation,
        selected_session_evaluations=selected_sessions,
        evaluated_epochs=evaluated_epochs,
    )


def train_pretrained_model(
    model_path: Path,
    training: list[Sample],
    validation: list[Sample],
    validation_sessions: list[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    learning_rate_schedule: str,
    minimum_learning_rate: float,
    trainable_scope: str,
    loss_formulation: str,
    augmentation_strength: str,
    weight_decay: float,
    evaluation_interval: int,
    seed: int,
    gates: QualityGates,
) -> TrainingOutcome:
    seed_everything(seed)
    device = execution_device()
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        GazeDataset(
            training, augment=True, augmentation_strength=augmentation_strength
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = openvino_gaze_model(model_path).to(device)
    configure_pretrained_trainable_scope(model, trainable_scope)
    optimiser = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = learning_rate_scheduler(
        optimiser, learning_rate_schedule, epochs, minimum_learning_rate
    )
    evaluated_epochs: list[EpochEvaluation] = []
    selected_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for left, right, pose, labels, _ in training_loader:
            optimiser.zero_grad(set_to_none=True)
            output = model(
                rgb_to_bgr(left.to(device)) * 255,
                rgb_to_bgr(right.to(device)) * 255,
                pose.to(device),
            ).reshape(-1, 3)
            output = functional.normalize(output, dim=1)
            target = openvino_target_vectors(labels.to(device))
            loss = aggregate_angular_training_loss(
                1 - torch.sum(output * target, dim=1), loss_formulation
            )
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch {epoch:03d} loss={statistics.fmean(losses):.5f}")
        if should_evaluate_epoch(epoch, epochs, evaluation_interval):
            sessions = evaluate_sessions(
                model, validation, validation_sessions, batch_size, device, True
            )
            result = epoch_evaluation(
                epoch, optimiser.param_groups[0]["lr"], sessions, gates
            )
            evaluated_epochs.append(result)
            if select_epoch(evaluated_epochs).epoch == epoch:
                selected_state = snapshot_model_state(model)
            print(f"epoch {epoch:03d} development_score={result.selection_score:.6f}")
        if scheduler is not None:
            scheduler.step()

    selected = select_epoch(evaluated_epochs)
    if selected_state is None:
        raise RuntimeError("selected model state was not retained")
    restore_model_state(model, selected_state)
    selected_sessions = evaluate_sessions(
        model, validation, validation_sessions, batch_size, device, True
    )
    selected_evaluation = evaluate_samples(model, validation, batch_size, device, True)
    model.cpu().eval()
    return TrainingOutcome(
        model=model,
        selected_epoch=selected.epoch,
        selected_evaluation=selected_evaluation,
        selected_session_evaluations=selected_sessions,
        evaluated_epochs=evaluated_epochs,
    )


def export_coreml(model: GazeEstimator, destination: Path) -> None:
    model.eval()
    example = (
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        torch.zeros(1, 3),
    )
    traced = torch.jit.trace(model, example)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.ImageType(
                name="left_eye",
                shape=example[0].shape,
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            ),
            ct.ImageType(
                name="right_eye",
                shape=example[1].shape,
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            ),
            ct.TensorType(name="head_pose", shape=example[2].shape),
        ],
        outputs=[ct.TensorType(name="gaze_degrees")],
    )
    if destination.exists():
        shutil.rmtree(destination)
    converted.save(destination)
    protect_private_artifact(destination)


def export_pretrained_coreml(model: nn.Module, destination: Path) -> None:
    class RGBInputAdapter(nn.Module):
        def __init__(self, source: nn.Module) -> None:
            super().__init__()
            self.source = source

        def forward(self, left: torch.Tensor, right: torch.Tensor,
                    pose: torch.Tensor) -> torch.Tensor:
            return self.source(rgb_to_bgr(left), rgb_to_bgr(right), pose)

    model = RGBInputAdapter(model).eval()
    example = (
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        torch.zeros(1, 3),
    )
    traced = torch.jit.trace(model, example)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.ImageType(
                name="left_eye",
                shape=example[0].shape,
                scale=1.0,
                color_layout=ct.colorlayout.RGB,
            ),
            ct.ImageType(
                name="right_eye",
                shape=example[1].shape,
                scale=1.0,
                color_layout=ct.colorlayout.RGB,
            ),
            ct.TensorType(name="head_pose", shape=example[2].shape),
        ],
        outputs=[ct.TensorType(name="gaze_vector")],
    )
    if destination.exists():
        shutil.rmtree(destination)
    converted.save(destination)
    protect_private_artifact(destination)


def model_digest(model: nn.Module) -> str:
    return state_dict_digest(model.state_dict())


def state_dict_digest(state: object) -> str:
    if not isinstance(state, dict) \
            or not state \
            or not all(isinstance(name, str) and isinstance(value, torch.Tensor)
                       for name, value in state.items()):
        raise ValueError("checkpoint model state is invalid")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} SHA-256 is invalid")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} SHA-256 is invalid") from error
    return value.lower()


def session_id_digest(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()


def sample_set_digest(samples: list[Sample]) -> str:
    digest = hashlib.sha256()
    image_digests: dict[Path, str] = {}
    for sample in sorted(
        samples, key=lambda value: (value.session_id, value.target_id, str(value.left_path))
    ):
        record = {
            "session_id": sample.session_id,
            "target_id": sample.target_id,
            "target_kind": sample.target_kind,
            "head_pose": sample.head_pose,
            "gaze_degrees": sample.gaze_degrees,
            "pose_prompt": sample.pose_prompt,
            "target_position": sample.target_position,
            "face_confidence": sample.face_confidence,
            "minimum_openness": sample.minimum_openness,
            "schema_version": sample.schema_version,
            "participant_binding": sample.participant_binding,
            "setup_binding": sample.setup_binding,
            "session_created_at": sample.session_created_at,
            "session_metadata_sha256": sample.session_metadata_sha256,
            "manifest_sha256": sample.manifest_sha256,
        }
        if sample.schema_version == 4:
            if sample.canonical_alignment is None:
                raise ValueError("schema-4 sample is missing canonical alignment evidence")
            record.update({
                "input_contract": sample_input_contract(sample),
                "label_contract_sha256": sample_label_contract_sha256(sample),
                "canonical_alignment": asdict(sample.canonical_alignment),
            })
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        for path in (sample.left_path, sample.right_path):
            if path not in image_digests:
                image_digests[path] = file_digest(path)
            digest.update(path.name.encode())
            digest.update(image_digests[path].encode())
    return digest.hexdigest()


def filtering_config(
    minimum_face_confidence: float = MINIMUM_FACE_CONFIDENCE,
) -> dict[str, float]:
    return {
        "minimum_face_confidence": minimum_face_confidence,
        "minimum_openness": MINIMUM_OPENNESS,
        "maximum_absolute_head_roll_degrees": MAXIMUM_ABSOLUTE_HEAD_ROLL_DEGREES,
        "minimum_validation_samples_per_pose": MINIMUM_VALIDATION_SAMPLES_PER_POSE,
        "minimum_validation_lens_samples": MINIMUM_VALIDATION_LENS_SAMPLES,
        "minimum_validation_lens_samples_per_pose": (
            MINIMUM_VALIDATION_LENS_SAMPLES_PER_POSE
        ),
        "minimum_validation_lens_targets_per_pose": (
            MINIMUM_VALIDATION_LENS_TARGETS_PER_POSE
        ),
        "minimum_validation_horizontal_pose_separation_degrees": (
            MINIMUM_VALIDATION_HORIZONTAL_POSE_SEPARATION_DEGREES
        ),
        "minimum_validation_vertical_pose_separation_degrees": (
            MINIMUM_VALIDATION_VERTICAL_POSE_SEPARATION_DEGREES
        ),
    }


def validated_filtering_config(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("frozen checkpoint filtering configuration is invalid")
    minimum_face_confidence = value.get("minimum_face_confidence")
    if isinstance(minimum_face_confidence, bool) \
            or not isinstance(minimum_face_confidence, (int, float)) \
            or not math.isfinite(minimum_face_confidence) \
            or not MINIMUM_FACE_CONFIDENCE <= minimum_face_confidence <= 1:
        raise ValueError("frozen checkpoint face-confidence filter is invalid")
    expected = filtering_config(float(minimum_face_confidence))
    if value != expected:
        raise ValueError("frozen checkpoint filtering configuration mismatch")
    return expected


def require_single_recording_setup(*sample_groups: Iterable[Sample]) -> str:
    bindings = {
        sample.setup_binding
        for samples in sample_groups
        for sample in samples
    }
    if len(bindings) != 1 or "" in bindings:
        raise ValueError(
            "personalized training, development and final sessions require one recording setup"
        )
    return next(iter(bindings))


def augmentation_config(strength: str) -> dict[str, object]:
    if strength not in AUGMENTATION_PROFILES:
        raise ValueError(f"unsupported augmentation strength: {strength}")
    return {
        "strength": strength,
        **AUGMENTATION_PROFILES[strength],
    }


def execution_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def hardware_model_identifier() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def software_versions(device: torch.device) -> dict[str, str]:
    packages = (
        "coremltools", "numpy", "onnx", "onnx2torch", "pillow", "torch", "torchvision"
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware_model": hardware_model_identifier(),
        "torch_device": device.type,
        "torch_mps_built": str(torch.backends.mps.is_built()),
        "torch_mps_available": str(torch.backends.mps.is_available()),
        "torch_deterministic_algorithms": str(
            torch.are_deterministic_algorithms_enabled()
        ),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def serialise_epoch(result: EpochEvaluation) -> dict[str, object]:
    return {
        "epoch": result.epoch,
        "learning_rate": result.learning_rate,
        "selection_score": result.selection_score,
        "worst_lens_ratio": result.worst_lens_ratio,
        "worst_p95_ratio": result.worst_p95_ratio,
        "worst_median_ratio": result.worst_median_ratio,
        "session_evaluations": {
            session_id: asdict(evaluation)
            for session_id, evaluation in result.session_evaluations.items()
        },
    }


def protect_private_artifact(path: Path) -> None:
    if path.is_dir():
        path.chmod(0o700)
        for child in path.rglob("*"):
            child.chmod(0o700 if child.is_dir() else 0o600)
    else:
        path.chmod(0o600)


def sync_to_permanent_storage(descriptor: int) -> None:
    command = getattr(fcntl, "F_FULLFSYNC", None)
    if command is None:
        os.fsync(descriptor)
    else:
        fcntl.fcntl(descriptor, command)


def durably_sync_published_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        sync_to_permanent_storage(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, directory_flags)
    try:
        sync_to_permanent_storage(descriptor)
    finally:
        os.close(descriptor)


def prepare_output_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def write_json_exclusive(
    path: Path, value: dict[str, object],
    before_publish: Callable[[], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"report already exists: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            sync_to_permanent_storage(handle.fileno())
        if before_publish is not None:
            before_publish()
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise ValueError(f"report already exists: {path}") from error
        protect_private_artifact(path)
        durably_sync_published_path(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint_model(
    checkpoint_path: Path, pretrained_onnx: Path | None
) -> tuple[nn.Module, dict, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError("unsupported checkpoint format")
    architecture = checkpoint.get("architecture")
    if architecture == PRETRAINED_ARCHITECTURE:
        if pretrained_onnx is None:
            raise ValueError("the pretrained ONNX source is required for this checkpoint")
        if checkpoint.get("pretrained_model_sha256") != OPENVINO_MODEL_SHA256:
            raise ValueError("checkpoint pretrained model checksum mismatch")
        model = openvino_gaze_model(pretrained_onnx)
        pretrained = True
    elif architecture == NATIVE_ARCHITECTURE:
        if pretrained_onnx is not None:
            raise ValueError("the native checkpoint does not use a pretrained ONNX source")
        model = GazeEstimator()
        pretrained = False
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    actual_digest = model_digest(model)
    if actual_digest != checkpoint.get("model_sha256"):
        raise ValueError("checkpoint model checksum mismatch")
    return model.eval(), checkpoint, pretrained


def gate_passes(evaluation: Evaluation, gates: QualityGates) -> bool:
    values = (
        evaluation.angular_median_degrees,
        evaluation.angular_p95_degrees,
        evaluation.lens_p95_degrees,
    )
    return all(math.isfinite(value) for value in values) and (
        evaluation.angular_median_degrees <= gates.maximum_median_degrees
        and evaluation.angular_p95_degrees <= gates.maximum_p95_degrees
        and evaluation.lens_p95_degrees <= gates.maximum_lens_p95_degrees
    )


def require_development_gate_pass(
    session_evaluations: dict[str, Evaluation],
    expected_sessions: Iterable[str],
    gates: QualityGates,
) -> None:
    expected = set(expected_sessions)
    if not expected or set(session_evaluations) != expected \
            or not all(gate_passes(value, gates) for value in session_evaluations.values()):
        raise ValueError("frozen checkpoint does not pass every development session")


def validated_recorded_development_evaluations(
    value: object, expected_sessions: Iterable[str], gates: QualityGates
) -> dict[str, Evaluation]:
    if not isinstance(value, dict):
        raise ValueError("frozen development evaluations are invalid")
    try:
        evaluations = {
            str(session_id): validated_evaluation(metrics)
            for session_id, metrics in value.items()
            if isinstance(metrics, dict)
        }
    except (TypeError, ValueError) as error:
        raise ValueError("frozen development evaluations are invalid") from error
    if len(evaluations) != len(value):
        raise ValueError("frozen development evaluations are invalid")
    selection_components(evaluations, gates)
    require_development_gate_pass(evaluations, expected_sessions, gates)
    return evaluations


def validated_evaluation(value: dict[str, object]) -> Evaluation:
    if set(value) != set(Evaluation.__dataclass_fields__):
        raise ValueError("evaluation fields are invalid")
    count = value["count"]
    lens_count = value["lens_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 \
            or isinstance(lens_count, bool) or not isinstance(lens_count, int) \
            or not 1 <= lens_count <= count:
        raise ValueError("evaluation counts are invalid")
    numeric_names = (
        "angular_median_degrees", "angular_p95_degrees", "lens_p95_degrees",
        "yaw_mae_degrees", "pitch_mae_degrees",
    )
    if any(
        isinstance(value[name], bool)
        or not isinstance(value[name], (int, float))
        or not math.isfinite(value[name])
        or value[name] < 0
        for name in numeric_names
    ):
        raise ValueError("evaluation metrics are invalid")
    return Evaluation(**value)


def read_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} is invalid: {path}")
    return value


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validated_session_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) \
            or not value \
            or not all(isinstance(session_id, str) and session_id for session_id in value) \
            or len(value) != len(set(value)):
        raise ValueError(f"{name} session list is invalid")
    return value


def candidate_procedure(report: dict[str, object]) -> dict[str, object]:
    try:
        return {
            factor: {name: report[name] for name in names}
            for factor, names in CANDIDATE_FACTOR_FIELDS.items()
        }
    except KeyError as error:
        raise ValueError("candidate report procedure is incomplete") from error


def candidate_shared_contract(report: dict[str, object]) -> dict[str, object]:
    try:
        return {name: report[name] for name in CANDIDATE_SHARED_FIELDS}
    except KeyError as error:
        raise ValueError("candidate report shared contract is incomplete") from error


def output_contract_for_architecture(architecture: object) -> str:
    if architecture == NATIVE_ARCHITECTURE:
        return "gaze_degrees"
    if architecture == PRETRAINED_ARCHITECTURE:
        return "gaze_vector"
    raise ValueError("gaze architecture is unsupported")


def validated_seed_run(report_path: Path) -> dict[str, object]:
    report_path = report_path.resolve()
    if report_path.name != "evaluation.json":
        raise ValueError("candidate inputs must be immutable evaluation.json reports")
    report = read_json_object(report_path, "candidate report")
    if report.get("schema_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("candidate reports must use the current checkpoint format")
    if report.get("output_contract") != output_contract_for_architecture(
        report.get("architecture")
    ):
        raise ValueError("candidate report output contract does not match its architecture")
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("candidate report seed is invalid")
    training_sessions = validated_session_list(
        report.get("training_sessions"), "training"
    )
    development_sessions = validated_session_list(
        report.get("development_sessions"), "development"
    )
    if len(development_sessions) < 2:
        raise ValueError("candidate reports require two development sessions")
    if set(training_sessions) & set(development_sessions):
        raise ValueError("candidate report data roles overlap")
    validated_completed_session_ledger(report.get("known_completed_sessions"))
    parsed_utc_timestamp(report.get("frozen_at"), "candidate report freeze")
    if report.get("trainer_sha256") != file_digest(Path(__file__).resolve()):
        raise ValueError("candidate report trainer differs from the current trainer")
    validated_filtering_config(report.get("filtering"))
    validated_input_preprocessing(report.get("input_preprocessing"))
    if validated_sha256(report.get("label_contract_sha256"), "label contract") \
            != label_contract_digest():
        raise ValueError("candidate report label contract is unsupported")
    for name in (
        "training_samples", "development_samples",
        "source_training_samples", "source_development_samples",
    ):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("candidate report sample counts are invalid")

    gates = quality_gates()
    value = report.get("session_evaluations")
    if not isinstance(value, dict):
        raise ValueError("candidate report development evaluations are invalid")
    try:
        evaluations = {
            str(session_id): validated_evaluation(metrics)
            for session_id, metrics in value.items()
            if isinstance(metrics, dict)
        }
    except (TypeError, ValueError) as error:
        raise ValueError("candidate report development evaluations are invalid") from error
    if len(evaluations) != len(value) \
            or set(evaluations) != set(development_sessions):
        raise ValueError("candidate report development evaluations are invalid")
    score, _, _, _ = selection_components(evaluations, gates)
    passed = all(gate_passes(evaluation, gates) for evaluation in evaluations.values())
    expected_gates = {**asdict(gates), "passed": passed}
    if report.get("gates") != expected_gates:
        raise ValueError("candidate report quality-gate result is invalid")
    selection = report.get("selection")
    selected_epoch = report.get("selected_epoch")
    evaluated_epochs = report.get("evaluated_epochs")
    if not isinstance(selection, dict) \
            or selection.get("selection_score") != score \
            or selection.get("session_evaluations") != value \
            or isinstance(selected_epoch, bool) \
            or not isinstance(selected_epoch, int) \
            or selected_epoch < 1 \
            or not isinstance(evaluated_epochs, list) \
            or not evaluated_epochs \
            or sum(
                isinstance(epoch, dict) and epoch.get("epoch") == selected_epoch
                for epoch in evaluated_epochs
            ) != 1 \
            or next(
                epoch for epoch in evaluated_epochs
                if isinstance(epoch, dict) and epoch.get("epoch") == selected_epoch
            ) != selection:
        raise ValueError("candidate report selection score is invalid")

    checkpoint_path = report_path.with_name("GazeEstimator.pt")
    if not checkpoint_path.is_file():
        raise ValueError(f"candidate checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = file_digest(checkpoint_path)
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("candidate checkpoint file checksum mismatch")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("candidate checkpoint is unreadable") from error
    if not isinstance(checkpoint, dict) \
            or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("candidate checkpoint format is invalid")
    model_sha256 = state_dict_digest(checkpoint.get("model_state"))
    if checkpoint.get("model_sha256") != model_sha256 \
            or report.get("model_sha256") != model_sha256:
        raise ValueError("candidate model checksum mismatch")
    mirrored_fields = (
        "architecture", "pretrained_model_sha256", "training_sessions",
        "development_sessions", "known_completed_sessions", "frozen_at",
        "setup_sha256", "training_data_sha256", "development_data_sha256",
        "source_training_data_sha256", "source_development_data_sha256",
        "source_training_samples", "source_development_samples",
        "training_epochs", "evaluation_interval", "selected_epoch", "seed",
        "evaluated_epochs", "selection", "batch_size", "trainable_parameters",
        "fine_tuning", "loss", "optimizer", "filtering", "augmentation",
        "input_preprocessing", "label_contract_sha256",
        "software_versions", "trainer_sha256",
    )
    if any(checkpoint.get(name) != report.get(name) for name in mirrored_fields) \
            or checkpoint.get("validation_sessions") != development_sessions \
            or checkpoint.get("quality_gates") != asdict(gates) \
            or checkpoint.get("development_passed") is not passed \
            or checkpoint.get("development_session_evaluations") != value:
        raise ValueError("candidate report and checkpoint contracts differ")
    procedure = candidate_procedure(report)
    shared = candidate_shared_contract(report)
    return {
        "seed": seed,
        "report": report,
        "report_path": report_path,
        "report_sha256": file_digest(report_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "model_sha256": model_sha256,
        "score": score,
        "passed": passed,
        "procedure": procedure,
        "shared": shared,
    }


def validated_seed_cohort(
    report_paths: Iterable[Path], name: str
) -> list[dict[str, object]]:
    paths = [path.resolve() for path in report_paths]
    if len(paths) != len(ROBUSTNESS_SEEDS) or len(paths) != len(set(paths)):
        raise ValueError(f"{name} cohort requires three distinct reports")
    runs = [validated_seed_run(path) for path in paths]
    seeds = [run["seed"] for run in runs]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(ROBUSTNESS_SEEDS):
        raise ValueError(f"{name} cohort must contain seeds {ROBUSTNESS_SEEDS}")
    runs.sort(key=lambda run: int(run["seed"]))
    reference_procedure = runs[0]["procedure"]
    reference_shared = runs[0]["shared"]
    if any(run["procedure"] != reference_procedure for run in runs[1:]):
        raise ValueError(f"{name} cohort procedure differs across seeds")
    if any(run["shared"] != reference_shared for run in runs[1:]):
        raise ValueError(f"{name} cohort data or environment differs across seeds")
    return runs


def relative_artifact_path(path: Path, manifest_path: Path) -> str:
    return os.path.relpath(path.resolve(), manifest_path.parent.resolve())


def seed_run_manifest_entry(
    run: dict[str, object], manifest_path: Path
) -> dict[str, object]:
    report_reference = relative_artifact_path(run["report_path"], manifest_path)
    checkpoint_reference = relative_artifact_path(
        run["checkpoint_path"], manifest_path
    )
    report = run["report"]
    session_ids = {
        session_id
        for name in (
            "training_sessions", "development_sessions", "known_completed_sessions"
        )
        for session_id in validated_session_list(report[name], name)
    }
    references = (report_reference.lower(), checkpoint_reference.lower())
    if any(
        session_id.lower() in reference
        for session_id in session_ids
        for reference in references
    ):
        raise ValueError("candidate artifact paths must not contain session identifiers")
    return {
        "seed": run["seed"],
        "report_path": report_reference,
        "report_sha256": run["report_sha256"],
        "checkpoint_path": checkpoint_reference,
        "checkpoint_sha256": run["checkpoint_sha256"],
        "model_sha256": run["model_sha256"],
        "selection_score": run["score"],
        "passed": run["passed"],
    }


def privacy_safe_shared_contract(shared: dict[str, object]) -> dict[str, object]:
    training_sessions = validated_session_list(
        shared["training_sessions"], "training"
    )
    development_sessions = validated_session_list(
        shared["development_sessions"], "development"
    )
    return {
        "training_session_set_sha256": canonical_digest(sorted(training_sessions)),
        "development_session_set_sha256": canonical_digest(sorted(development_sessions)),
        "source_training_data_sha256": shared["source_training_data_sha256"],
        "source_development_data_sha256": shared["source_development_data_sha256"],
        "source_training_samples": shared["source_training_samples"],
        "source_development_samples": shared["source_development_samples"],
        "setup_sha256": shared["setup_sha256"],
        "label_contract_sha256": shared["label_contract_sha256"],
        "completed_session_set_sha256": canonical_digest(
            sorted(shared["known_completed_sessions"])
        ),
        "software_versions": shared["software_versions"],
        "trainer_sha256": shared["trainer_sha256"],
    }


def build_candidate_manifest(
    candidate_report_paths: Iterable[Path],
    baseline_report_paths: Iterable[Path],
    changed_factor: str,
    known_completed_session_sha256: Iterable[str],
    output_path: Path,
    frozen_at: str,
) -> dict[str, object]:
    candidate_runs = validated_seed_cohort(candidate_report_paths, "candidate")
    baseline_runs = validated_seed_cohort(baseline_report_paths, "baseline")
    return build_candidate_manifest_from_runs(
        candidate_runs,
        baseline_runs,
        changed_factor,
        known_completed_session_sha256,
        output_path,
        frozen_at,
    )


def build_candidate_manifest_from_runs(
    candidate_runs: list[dict[str, object]],
    baseline_runs: list[dict[str, object]],
    changed_factor: str,
    known_completed_session_sha256: Iterable[str],
    output_path: Path,
    frozen_at: str,
) -> dict[str, object]:
    if changed_factor not in CANDIDATE_FACTOR_FIELDS:
        raise ValueError("candidate changed factor is invalid")
    parsed_utc_timestamp(frozen_at, "candidate manifest freeze")
    candidate_shared = candidate_runs[0]["shared"]
    baseline_shared = baseline_runs[0]["shared"]
    if candidate_shared != baseline_shared:
        raise ValueError("candidate and baseline data or environments differ")
    candidate_procedure_value = candidate_runs[0]["procedure"]
    baseline_procedure_value = baseline_runs[0]["procedure"]
    differing_factors = {
        factor for factor in CANDIDATE_FACTOR_FIELDS
        if candidate_procedure_value[factor] != baseline_procedure_value[factor]
    }
    if differing_factors != {changed_factor}:
        raise ValueError(
            "candidate differs from baseline outside the declared major factor"
        )
    if not all(bool(run["passed"]) for run in candidate_runs):
        raise ValueError("every candidate seed must pass every development session")
    candidate_by_seed = {int(run["seed"]): run for run in candidate_runs}
    baseline_by_seed = {int(run["seed"]): run for run in baseline_runs}
    paired_scores = []
    for seed in ROBUSTNESS_SEEDS:
        candidate_score = float(candidate_by_seed[seed]["score"])
        baseline_score = float(baseline_by_seed[seed]["score"])
        if not candidate_score < baseline_score:
            raise ValueError("candidate improvement is not consistent across seeds")
        paired_scores.append({
            "seed": seed,
            "baseline_selection_score": baseline_score,
            "candidate_selection_score": candidate_score,
            "improvement": baseline_score - candidate_score,
        })
    known_hashes = [
        validated_sha256(value, "completed session")
        for value in known_completed_session_sha256
    ]
    if not known_hashes or len(known_hashes) != len(set(known_hashes)):
        raise ValueError("candidate completed-session snapshot is invalid")
    known_hashes.sort()
    latest_report_freeze = max(
        parsed_utc_timestamp(run["report"]["frozen_at"], "candidate report freeze")
        for run in candidate_runs + baseline_runs
    )
    manifest_freeze = parsed_utc_timestamp(frozen_at, "candidate manifest freeze")
    if manifest_freeze < latest_report_freeze:
        raise ValueError("candidate manifest predates a robustness run")
    candidate_scores = [float(run["score"]) for run in candidate_runs]
    baseline_scores = [float(run["score"]) for run in baseline_runs]
    primary = candidate_by_seed[PRIMARY_CANDIDATE_SEED]
    shared_without_identifiers = privacy_safe_shared_contract(candidate_shared)
    candidate_procedure_sha256 = canonical_digest(candidate_procedure_value)
    baseline_procedure_sha256 = canonical_digest(baseline_procedure_value)
    semantic_shared_contract = {
        name: value for name, value in shared_without_identifiers.items()
        if name != "completed_session_set_sha256"
    }
    candidate_identity_sha256 = canonical_digest({
        "schema_version": CANDIDATE_MANIFEST_FORMAT_VERSION,
        "seeds": list(ROBUSTNESS_SEEDS),
        "primary_seed": PRIMARY_CANDIDATE_SEED,
        "shared_contract_sha256": canonical_digest(semantic_shared_contract),
        "candidate_procedure_sha256": candidate_procedure_sha256,
        "candidate_models": [
            {"seed": run["seed"], "model_sha256": run["model_sha256"]}
            for run in candidate_runs
        ],
    })
    return {
        "schema_version": CANDIDATE_MANIFEST_FORMAT_VERSION,
        "frozen_at": frozen_at,
        "seeds": list(ROBUSTNESS_SEEDS),
        "primary_seed": PRIMARY_CANDIDATE_SEED,
        "changed_factor": changed_factor,
        "known_completed_session_sha256": known_hashes,
        "shared_contract_sha256": canonical_digest(shared_without_identifiers),
        "software_versions": candidate_shared["software_versions"],
        "trainer_sha256": candidate_shared["trainer_sha256"],
        "candidate_procedure": candidate_procedure_value,
        "baseline_procedure": baseline_procedure_value,
        "candidate_procedure_sha256": candidate_procedure_sha256,
        "baseline_procedure_sha256": baseline_procedure_sha256,
        "candidate_identity_sha256": candidate_identity_sha256,
        "candidate_runs": [
            seed_run_manifest_entry(run, output_path) for run in candidate_runs
        ],
        "baseline_runs": [
            seed_run_manifest_entry(run, output_path) for run in baseline_runs
        ],
        "paired_scores": paired_scores,
        "robustness": {
            "passed": True,
            "candidate_worst_selection_score": max(candidate_scores),
            "candidate_median_selection_score": statistics.median(candidate_scores),
            "baseline_worst_selection_score": max(baseline_scores),
            "minimum_paired_improvement": min(
                pair["improvement"] for pair in paired_scores
            ),
        },
        "primary_checkpoint_sha256": primary["checkpoint_sha256"],
        "primary_model_sha256": primary["model_sha256"],
    }


def manifest_report_paths(
    manifest: dict[str, object], name: str, manifest_path: Path
) -> list[Path]:
    entries = manifest.get(name)
    if not isinstance(entries, list) or len(entries) != len(ROBUSTNESS_SEEDS) \
            or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("candidate manifest run list is invalid")
    paths = []
    for entry in entries:
        value = entry.get("report_path")
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("candidate manifest report path is invalid")
        paths.append((manifest_path.parent / value).resolve())
    return paths


def require_frozen_candidate_manifest(
    checkpoint_path: Path, expected_checkpoint_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    manifest_path = checkpoint_path.with_name(CANDIDATE_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        raise ValueError(f"frozen candidate manifest is missing: {manifest_path}")
    if file_digest(manifest_path) != validated_sha256(
        expected_manifest_sha256, "candidate manifest"
    ):
        raise ValueError("frozen candidate manifest checksum mismatch")
    manifest = read_json_object(manifest_path, "candidate manifest")
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_FORMAT_VERSION \
            or manifest.get("seeds") != list(ROBUSTNESS_SEEDS) \
            or manifest.get("primary_seed") != PRIMARY_CANDIDATE_SEED:
        raise ValueError("frozen candidate manifest contract is invalid")
    known_hashes = manifest.get("known_completed_session_sha256")
    if not isinstance(known_hashes, list):
        raise ValueError("frozen candidate completed-session snapshot is invalid")
    rebuilt = build_candidate_manifest(
        manifest_report_paths(manifest, "candidate_runs", manifest_path),
        manifest_report_paths(manifest, "baseline_runs", manifest_path),
        str(manifest.get("changed_factor")),
        known_hashes,
        manifest_path,
        str(manifest.get("frozen_at")),
    )
    if rebuilt != manifest:
        raise ValueError("frozen candidate manifest does not match its artifacts")
    expected_checkpoint_sha256 = validated_sha256(
        expected_checkpoint_sha256, "checkpoint"
    )
    if manifest.get("primary_checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("checkpoint is not the frozen seed-7 candidate")
    primary_entries = [
        entry for entry in manifest["candidate_runs"]
        if entry.get("seed") == PRIMARY_CANDIDATE_SEED
    ]
    if len(primary_entries) != 1:
        raise ValueError("frozen candidate primary run is invalid")
    primary_checkpoint = manifest_path.parent / primary_entries[0]["checkpoint_path"]
    if primary_checkpoint.resolve() != checkpoint_path.resolve():
        raise ValueError("checkpoint path is not the frozen seed-7 candidate")
    return manifest


def run_freeze_candidate(args: argparse.Namespace) -> int:
    candidate_reports = [path.resolve() for path in args.candidate_report]
    candidate_runs = validated_seed_cohort(candidate_reports, "candidate")
    baseline_runs = validated_seed_cohort(args.baseline_report, "baseline")
    primary = next(
        run for run in candidate_runs if run["seed"] == PRIMARY_CANDIDATE_SEED
    )
    required_output = primary["report_path"].with_name(CANDIDATE_MANIFEST_FILENAME)
    if args.output.resolve() != required_output.resolve():
        raise ValueError(f"candidate manifest output must be {required_output}")
    initial_completed_sessions = finished_session_ids(args.data)
    manifest = build_candidate_manifest_from_runs(
        candidate_runs,
        baseline_runs,
        args.changed_factor,
        [session_id_digest(value) for value in initial_completed_sessions],
        args.output,
        datetime.now(timezone.utc).isoformat(),
    )
    manifest["frozen_at"] = datetime.now(timezone.utc).isoformat()
    write_json_exclusive(
        args.output,
        manifest,
        before_publish=lambda: require_stable_completed_session_ledger(
            initial_completed_sessions,
            finished_session_ids(args.data),
            operation="candidate freeze",
        ),
    )
    print(json.dumps({
        "candidate_manifest_sha256": file_digest(args.output),
        "primary_checkpoint_sha256": manifest["primary_checkpoint_sha256"],
        "robustness": manifest["robustness"],
    }, indent=2, sort_keys=True))
    return 0


def validate_training_arguments(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.evaluation_interval < 1:
        raise ValueError("epochs, batch size and evaluation interval must be positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    if not math.isfinite(args.minimum_face_confidence) \
            or not MINIMUM_FACE_CONFIDENCE <= args.minimum_face_confidence <= 1:
        raise ValueError(
            f"minimum face confidence must be between {MINIMUM_FACE_CONFIDENCE} and one"
        )
    if args.learning_rate is not None \
            and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.minimum_learning_rate) \
            or args.minimum_learning_rate < 0:
        raise ValueError("minimum learning rate must be finite and nonnegative")


def run_train(args: argparse.Namespace) -> int:
    validate_training_arguments(args)
    initial_completed_sessions = finished_session_ids(args.data)
    training_ids = set(args.training_session_id)
    validation_ids = set(args.validation_session_id)
    training, training_participants, training_sessions = load_samples(
        args.data, "training", training_ids,
        minimum_face_confidence=args.minimum_face_confidence,
    )
    validation, validation_participants, validation_sessions = load_samples(
        args.data, "validation", validation_ids,
        minimum_face_confidence=args.minimum_face_confidence,
    )
    if args.minimum_face_confidence == MINIMUM_FACE_CONFIDENCE:
        source_training = training
        source_validation = validation
    else:
        source_training, source_training_participants, source_training_sessions = load_samples(
            args.data, "training", training_ids,
            minimum_face_confidence=MINIMUM_FACE_CONFIDENCE,
        )
        (
            source_validation,
            source_validation_participants,
            source_validation_sessions,
        ) = load_samples(
            args.data,
            "validation",
            validation_ids,
            minimum_face_confidence=MINIMUM_FACE_CONFIDENCE,
        )
        if source_training_participants != training_participants \
                or source_validation_participants != validation_participants \
                or source_training_sessions != training_sessions \
                or source_validation_sessions != validation_sessions:
            raise ValueError("source and retained dataset roles differ")
        validate_pose_coverage(source_validation, source_validation_sessions)
        validate_numeric_pitch_contract(
            source_training, source_training_sessions, require_head_pitch=False
        )
        validate_numeric_pitch_contract(
            source_validation, source_validation_sessions
        )
    validate_pose_coverage(validation, validation_sessions)
    validate_numeric_pitch_contract(
        training, training_sessions, require_head_pitch=False
    )
    validate_numeric_pitch_contract(validation, validation_sessions)
    participants = training_participants | validation_participants
    if len(participants) != 1 or training_participants != validation_participants:
        raise ValueError(
            "the personalized gate requires training and validation from one participant"
        )
    if set(training_sessions) & set(validation_sessions):
        raise ValueError("training and validation sessions overlap")
    input_preprocessing = require_single_input_preprocessing(
        source_training, source_validation, training, validation
    )
    label_contract_sha256 = require_single_label_contract(
        source_training, source_validation, training, validation
    )
    setup_sha256 = require_single_recording_setup(training, validation)

    pretrained = args.pretrained_onnx is not None
    if not pretrained and args.pretrained_trainable_scope != "all":
        raise ValueError("pretrained trainable scope requires a pretrained model")
    if not pretrained and args.pretrained_loss != "mean-angular":
        raise ValueError("pretrained loss formulation requires a pretrained model")
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-4 if pretrained else 1e-3
    if args.learning_rate_schedule != "constant" and (
        args.minimum_learning_rate >= learning_rate
    ):
        raise ValueError("minimum learning rate must be nonnegative and below the initial rate")
    prepare_output_directory(args.output)
    gates = quality_gates()
    if pretrained:
        outcome = train_pretrained_model(
            args.pretrained_onnx,
            training,
            validation,
            validation_sessions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            learning_rate_schedule=args.learning_rate_schedule,
            minimum_learning_rate=args.minimum_learning_rate,
            trainable_scope=args.pretrained_trainable_scope,
            loss_formulation=args.pretrained_loss,
            augmentation_strength=args.augmentation_strength,
            weight_decay=args.weight_decay,
            evaluation_interval=args.evaluation_interval,
            seed=args.seed,
            gates=gates,
        )
    else:
        outcome = train_model(
            training,
            validation,
            validation_sessions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            learning_rate_schedule=args.learning_rate_schedule,
            minimum_learning_rate=args.minimum_learning_rate,
            augmentation_strength=args.augmentation_strength,
            weight_decay=args.weight_decay,
            evaluation_interval=args.evaluation_interval,
            seed=args.seed,
            gates=gates,
        )
    model = outcome.model
    architecture = PRETRAINED_ARCHITECTURE if pretrained else NATIVE_ARCHITECTURE
    output_contract = output_contract_for_architecture(architecture)
    diagnostics_device = execution_device()
    model = model.to(diagnostics_device)
    diagnostics = diagnostic_breakdowns(
        model, validation, validation_sessions, args.batch_size, diagnostics_device, pretrained
    )
    model.cpu().eval()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters()
                               if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if not pretrained and total_parameters != NATIVE_PARAMETER_COUNT:
        raise RuntimeError("pinned native model parameter count changed")
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    fine_tuning = {
        "scope": args.pretrained_trainable_scope if pretrained else "all",
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_names": trainable_parameter_names,
    }
    loss_configuration = training_loss_configuration(pretrained, args.pretrained_loss)
    optimiser = {
        "name": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.learning_rate_schedule,
        "minimum_learning_rate": (
            args.minimum_learning_rate
            if args.learning_rate_schedule != "constant" else None
        ),
        "scheduler_step_unit": (
            "epoch" if args.learning_rate_schedule != "constant" else None
        ),
    }
    versions = software_versions(diagnostics_device)
    trainer_sha256 = file_digest(Path(__file__).resolve())
    training_data_sha256 = sample_set_digest(training)
    development_data_sha256 = sample_set_digest(validation)
    source_training_data_sha256 = sample_set_digest(source_training)
    source_development_data_sha256 = sample_set_digest(source_validation)
    evaluated_epochs = [serialise_epoch(result) for result in outcome.evaluated_epochs]
    selected = next(
        result for result in outcome.evaluated_epochs
        if result.epoch == outcome.selected_epoch
    )
    passed = all(
        gate_passes(evaluation, gates)
        for evaluation in outcome.selected_session_evaluations.values()
    )
    filtering = filtering_config(args.minimum_face_confidence)
    known_completed_sessions = require_stable_completed_session_ledger(
        initial_completed_sessions, finished_session_ids(args.data)
    )
    frozen_at = datetime.now(timezone.utc).isoformat()
    checkpoint_payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "model_state": model.state_dict(),
        "model_sha256": model_digest(model),
        "pretrained_model_sha256": OPENVINO_MODEL_SHA256 if pretrained else None,
        "training_sessions": training_sessions,
        "development_sessions": validation_sessions,
        "validation_sessions": validation_sessions,
        "known_completed_sessions": known_completed_sessions,
        "frozen_at": frozen_at,
        "setup_sha256": setup_sha256,
        "training_data_sha256": training_data_sha256,
        "development_data_sha256": development_data_sha256,
        "source_training_data_sha256": source_training_data_sha256,
        "source_development_data_sha256": source_development_data_sha256,
        "source_training_samples": len(source_training),
        "source_development_samples": len(source_validation),
        "training_epochs": args.epochs,
        "evaluation_interval": args.evaluation_interval,
        "selected_epoch": outcome.selected_epoch,
        "evaluated_epochs": evaluated_epochs,
        "selection": serialise_epoch(selected),
        "development_passed": passed,
        "development_session_evaluations": {
            session_id: asdict(evaluation)
            for session_id, evaluation in outcome.selected_session_evaluations.items()
        },
        "quality_gates": asdict(gates),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "trainable_parameters": trainable_parameters,
        "fine_tuning": fine_tuning,
        "loss": loss_configuration,
        "optimizer": optimiser,
        "filtering": filtering,
        "augmentation": augmentation_config(args.augmentation_strength),
        "input_preprocessing": input_preprocessing,
        "label_contract_sha256": label_contract_sha256,
        "software_versions": versions,
        "trainer_sha256": trainer_sha256,
    }
    checkpoint = args.output / "GazeEstimator.pt"
    torch.save(checkpoint_payload, checkpoint)
    protect_private_artifact(checkpoint)
    checkpoint_sha256 = file_digest(checkpoint)
    report = {
        "schema_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "output_contract": output_contract,
        "checkpoint_sha256": checkpoint_sha256,
        "model_sha256": model_digest(model),
        "pretrained_model_sha256": OPENVINO_MODEL_SHA256 if pretrained else None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "training_epochs": args.epochs,
        "evaluation_interval": args.evaluation_interval,
        "selected_epoch": outcome.selected_epoch,
        "evaluated_epochs": evaluated_epochs,
        "selection": serialise_epoch(selected),
        "trainable_parameters": trainable_parameters,
        "fine_tuning": fine_tuning,
        "loss": loss_configuration,
        "optimizer": optimiser,
        "filtering": filtering,
        "augmentation": augmentation_config(args.augmentation_strength),
        "input_preprocessing": input_preprocessing,
        "label_contract_sha256": label_contract_sha256,
        "software_versions": versions,
        "trainer_sha256": trainer_sha256,
        "frozen_at": frozen_at,
        "training_data_sha256": training_data_sha256,
        "development_data_sha256": development_data_sha256,
        "source_training_data_sha256": source_training_data_sha256,
        "source_development_data_sha256": source_development_data_sha256,
        "training_samples": len(training),
        "development_samples": len(validation),
        "source_training_samples": len(source_training),
        "source_development_samples": len(source_validation),
        "training_sessions": training_sessions,
        "development_sessions": validation_sessions,
        "known_completed_sessions": known_completed_sessions,
        "setup_sha256": setup_sha256,
        "evaluation": asdict(outcome.selected_evaluation),
        "session_evaluations": {
            session_id: asdict(evaluation)
            for session_id, evaluation in outcome.selected_session_evaluations.items()
        },
        "diagnostics": diagnostics,
        "gates": {
            **asdict(gates),
            "passed": passed,
        },
    }
    write_json_exclusive(args.output / "evaluation.json", report)
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))
    if not passed:
        print("quality gate failed; checkpoint retained but Core ML export withheld", file=sys.stderr)
        return 2
    model_path = args.output / "GazeEstimator.mlpackage"
    if pretrained:
        export_pretrained_coreml(model, model_path)
    else:
        export_coreml(model, model_path)
    print(f"quality gate passed; exported {model_path}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    require_final_evaluation_output(args.checkpoint, args.output)
    if args.output.exists():
        raise ValueError(f"evaluation report already exists: {args.output}")
    actual_checkpoint_sha256 = file_digest(args.checkpoint)
    if actual_checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("frozen checkpoint file checksum mismatch")
    model, checkpoint, pretrained = load_checkpoint_model(
        args.checkpoint, args.pretrained_onnx
    )
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"final evaluation requires a version-{CHECKPOINT_FORMAT_VERSION} frozen checkpoint"
        )
    if checkpoint.get("trainer_sha256") != file_digest(Path(__file__).resolve()):
        raise ValueError("final evaluator differs from the frozen trainer")
    candidate_manifest = require_frozen_candidate_manifest(
        args.checkpoint,
        actual_checkpoint_sha256,
        args.expected_candidate_manifest_sha256,
    )
    candidate_manifest_sha256 = file_digest(
        args.checkpoint.with_name(CANDIDATE_MANIFEST_FILENAME)
    )
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("frozen checkpoint seed is invalid")
    seed_everything(seed)
    device = execution_device()
    versions = software_versions(device)
    if checkpoint.get("software_versions") != versions:
        raise ValueError("final evaluation software differs from the frozen environment")
    if checkpoint.get("batch_size") != args.batch_size:
        raise ValueError("final evaluation batch size differs from the frozen checkpoint")
    filtering = validated_filtering_config(checkpoint.get("filtering"))
    minimum_face_confidence = filtering["minimum_face_confidence"]
    validation_ids = set(args.validation_session_id)
    if len(validation_ids) != 1:
        raise ValueError("final evaluation requires exactly one untouched session")
    known_completed_session_sha256 = candidate_manifest[
        "known_completed_session_sha256"
    ]
    current_completed_sessions = finished_session_ids(args.data)
    require_single_new_completed_session_hashes(
        validation_ids,
        known_completed_session_sha256,
        current_completed_sessions,
    )
    validation, validation_participants, validation_sessions = load_samples(
        args.data, "validation", validation_ids,
        minimum_face_confidence=minimum_face_confidence,
    )
    validate_pose_coverage(validation, validation_sessions)
    validate_numeric_pitch_contract(validation, validation_sessions)
    training_sessions = checkpoint.get("training_sessions")
    if not isinstance(training_sessions, list) or not training_sessions:
        raise ValueError("checkpoint does not identify its training sessions")
    development_sessions = checkpoint.get("development_sessions")
    if not isinstance(development_sessions, list) or len(development_sessions) < 2:
        raise ValueError("final evaluation requires two frozen development sessions")
    training, training_participants, loaded_training_sessions = load_samples(
        args.data, "training", set(training_sessions),
        minimum_face_confidence=minimum_face_confidence,
    )
    development, development_participants, loaded_development_sessions = load_samples(
        args.data, "validation", set(development_sessions),
        minimum_face_confidence=minimum_face_confidence,
    )
    validate_pose_coverage(development, loaded_development_sessions)
    validate_numeric_pitch_contract(
        training, loaded_training_sessions, require_head_pitch=False
    )
    validate_numeric_pitch_contract(development, loaded_development_sessions)
    if minimum_face_confidence == MINIMUM_FACE_CONFIDENCE:
        source_training = training
        source_development = development
    else:
        source_training, source_training_participants, source_training_sessions = load_samples(
            args.data, "training", set(training_sessions),
            minimum_face_confidence=MINIMUM_FACE_CONFIDENCE,
        )
        (
            source_development,
            source_development_participants,
            source_development_sessions,
        ) = load_samples(
            args.data,
            "validation",
            set(development_sessions),
            minimum_face_confidence=MINIMUM_FACE_CONFIDENCE,
        )
        if source_training_participants != training_participants \
                or source_development_participants != development_participants \
                or source_training_sessions != loaded_training_sessions \
                or source_development_sessions != loaded_development_sessions:
            raise ValueError("source and retained dataset roles differ")
        validate_pose_coverage(source_development, source_development_sessions)
        validate_numeric_pitch_contract(
            source_training, source_training_sessions, require_head_pitch=False
        )
        validate_numeric_pitch_contract(
            source_development, source_development_sessions
        )
    if len(validation_participants) != 1 \
            or training_participants != validation_participants \
            or development_participants != validation_participants:
        raise ValueError("the personalized gate requires validation from the training participant")
    input_preprocessing = require_single_input_preprocessing(
        source_training, source_development, training, development, validation
    )
    frozen_input_preprocessing = validated_input_preprocessing(
        checkpoint.get("input_preprocessing")
    )
    if frozen_input_preprocessing != input_preprocessing:
        raise ValueError("final input preprocessing differs from the frozen checkpoint")
    label_contract_sha256 = require_single_label_contract(
        source_training, source_development, training, development, validation
    )
    if checkpoint.get("label_contract_sha256") != label_contract_sha256:
        raise ValueError("final label contract differs from the frozen checkpoint")
    setup_sha256 = require_single_recording_setup(
        training, development, validation
    )
    if checkpoint.get("setup_sha256") != setup_sha256:
        raise ValueError("final recording setup differs from the frozen checkpoint")
    require_untouched_final_session(
        validation_sessions, training_sessions, development_sessions
    )
    require_final_session_created_after_freeze(
        validation, candidate_manifest.get("frozen_at")
    )
    if checkpoint.get("training_data_sha256") != sample_set_digest(training):
        raise ValueError("frozen training dataset checksum mismatch")
    if checkpoint.get("development_data_sha256") != sample_set_digest(development):
        raise ValueError("frozen development dataset checksum mismatch")
    if checkpoint.get("source_training_data_sha256") != sample_set_digest(source_training) \
            or checkpoint.get("source_training_samples") != len(source_training):
        raise ValueError("frozen source training dataset checksum mismatch")
    if checkpoint.get("source_development_data_sha256") \
            != sample_set_digest(source_development) \
            or checkpoint.get("source_development_samples") != len(source_development):
        raise ValueError("frozen source development dataset checksum mismatch")
    gates = quality_gates()
    if checkpoint.get("quality_gates") != asdict(gates):
        raise ValueError("final quality gates differ from the frozen checkpoint")
    if checkpoint.get("development_passed") is not True:
        raise ValueError("checkpoint was not frozen after a development pass")
    recorded_development_evaluations = validated_recorded_development_evaluations(
        checkpoint.get("development_session_evaluations"),
        loaded_development_sessions, gates,
    )

    expected_digest = model_digest(model)
    model = model.to(device)
    development_session_evaluations = evaluate_sessions(
        model, development, loaded_development_sessions,
        args.batch_size, device, pretrained,
    )
    require_development_gate_pass(
        development_session_evaluations, loaded_development_sessions, gates
    )
    if {
        session_id: asdict(value)
        for session_id, value in development_session_evaluations.items()
    } != {
        session_id: asdict(value)
        for session_id, value in recorded_development_evaluations.items()
    }:
        raise ValueError("replayed development metrics differ from the frozen checkpoint")
    final_session_claim_sha256 = claim_final_evaluation_session(
        final_evaluation_ledger_root(),
        validation_sessions[0],
        candidate_manifest["candidate_identity_sha256"],
        actual_checkpoint_sha256,
    )
    session_evaluations = evaluate_sessions(
        model, validation, validation_sessions, args.batch_size, device, pretrained
    )
    evaluation = evaluate_samples(model, validation, args.batch_size, device, pretrained)
    diagnostics = diagnostic_breakdowns(
        model, validation, validation_sessions, args.batch_size, device, pretrained
    )
    model.cpu().eval()
    if model_digest(model) != expected_digest:
        raise RuntimeError("evaluation changed the frozen model weights")

    passed = all(gate_passes(value, gates) for value in session_evaluations.values())
    report = {
        "schema_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": checkpoint["architecture"],
        "checkpoint_sha256": actual_checkpoint_sha256,
        "model_sha256": expected_digest,
        "pretrained_model_sha256": checkpoint.get("pretrained_model_sha256"),
        "training_sessions": loaded_training_sessions,
        "development_sessions": loaded_development_sessions,
        "known_completed_session_sha256": known_completed_session_sha256,
        "setup_sha256": setup_sha256,
        "checkpoint_frozen_at": checkpoint["frozen_at"],
        "candidate_frozen_at": candidate_manifest["frozen_at"],
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_identity_sha256": candidate_manifest[
            "candidate_identity_sha256"
        ],
        "candidate_robustness": candidate_manifest["robustness"],
        "final_session_claim_sha256": final_session_claim_sha256,
        "seed": checkpoint["seed"],
        "batch_size": checkpoint["batch_size"],
        "optimizer": checkpoint["optimizer"],
        "fine_tuning": checkpoint["fine_tuning"],
        "loss": checkpoint["loss"],
        "augmentation": checkpoint["augmentation"],
        "input_preprocessing": input_preprocessing,
        "label_contract_sha256": label_contract_sha256,
        "training_software_versions": checkpoint["software_versions"],
        "trainer_sha256": checkpoint["trainer_sha256"],
        "selected_epoch": checkpoint["selected_epoch"],
        "selection": checkpoint["selection"],
        "evaluated_epochs": checkpoint["evaluated_epochs"],
        "training_data_sha256": checkpoint["training_data_sha256"],
        "development_data_sha256": checkpoint["development_data_sha256"],
        "source_training_data_sha256": checkpoint["source_training_data_sha256"],
        "source_development_data_sha256": checkpoint[
            "source_development_data_sha256"
        ],
        "source_training_samples": len(source_training),
        "source_development_samples": len(source_development),
        "validation_data_sha256": sample_set_digest(validation),
        "validation_samples": len(validation),
        "validation_sessions": validation_sessions,
        "frozen_model_unchanged": True,
        "evaluation": asdict(evaluation),
        "session_evaluations": {
            session_id: asdict(value)
            for session_id, value in session_evaluations.items()
        },
        "development_session_evaluations": {
            session_id: asdict(value)
            for session_id, value in development_session_evaluations.items()
        },
        "diagnostics": diagnostics,
        "filtering": filtering,
        "software_versions": versions,
        "gates": {
            **asdict(gates),
            "passed": passed,
        },
    }
    write_json_exclusive(
        args.output,
        report,
        before_publish=lambda: require_stable_completed_session_ledger(
            current_completed_sessions,
            finished_session_ids(args.data),
            operation="final evaluation",
        ),
    )
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))
    if not passed:
        print("final quality gate failed; frozen checkpoint unchanged", file=sys.stderr)
        return 2
    print("final quality gate passed; frozen checkpoint unchanged")
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    pretrained = args.pretrained_onnx is not None
    model = (openvino_gaze_model(args.pretrained_onnx)
             if pretrained else GazeEstimator()).eval()
    with tempfile.TemporaryDirectory(prefix="aspectus-gaze-smoke-") as directory:
        destination = Path(directory) / "GazeEstimator.mlpackage"
        if pretrained:
            export_pretrained_coreml(model, destination)
        else:
            export_coreml(model, destination)
        specification = ct.utils.load_spec(str(destination))
        inputs = [feature.name for feature in specification.description.input]
        outputs = [feature.name for feature in specification.description.output]
        expected_output = "gaze_vector" if pretrained else "gaze_degrees"
        if inputs != ["left_eye", "right_eye", "head_pose"] or outputs != [expected_output]:
            raise RuntimeError(f"unexpected Core ML contract: inputs={inputs}, outputs={outputs}")
        generator = np.random.default_rng(args.seed)
        left = Image.fromarray(generator.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        right = Image.fromarray(generator.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        head_pose = np.array([[8.0, -5.0, 2.0]], dtype=np.float32)
        with torch.inference_mode():
            left_tensor = image_tensor(left).unsqueeze(0)
            right_tensor = image_tensor(right).unsqueeze(0)
            if pretrained:
                left_tensor = rgb_to_bgr(left_tensor) * 255
                right_tensor = rgb_to_bgr(right_tensor) * 255
            expected = model(left_tensor, right_tensor,
                             torch.from_numpy(head_pose)).numpy().reshape(-1)
        runtime = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_ONLY)
        actual = np.asarray(
            runtime.predict(
                {"left_eye": left, "right_eye": right, "head_pose": head_pose}
            )[expected_output]
        ).reshape(-1)
        if pretrained:
            actual_degrees = openvino_vectors_to_degrees(
                torch.from_numpy(actual).reshape(1, 3)
            ).numpy()
            expected_degrees = openvino_vectors_to_degrees(
                torch.from_numpy(expected).reshape(1, 3)
            ).numpy()
            maximum_difference = float(
                angular_errors(actual_degrees, expected_degrees)[0]
            )
            tolerance = MAXIMUM_COREML_ANGULAR_DIFFERENCE_DEGREES
            difference_label = "angular difference"
        else:
            maximum_difference = float(np.max(np.abs(actual - expected)))
            tolerance = 0.05
            difference_label = "degree-output difference"
        if maximum_difference > tolerance:
            raise RuntimeError(
                f"Core ML {difference_label} from PyTorch is {maximum_difference:.4f}"
            )
        print(
            f"conversion smoke passed: {sum(parameter.numel() for parameter in model.parameters())} "
            f"parameters, max {difference_label}={maximum_difference:.4f}, "
            f"inputs={inputs}, outputs={outputs}"
        )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="convert an untrained model and verify its contract")
    smoke.add_argument("--seed", type=int, default=7)
    smoke.add_argument("--pretrained-onnx", type=Path)
    smoke.set_defaults(action=run_smoke)

    train = subparsers.add_parser("train", help="train on complete sessions and evaluate held out")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float)
    train.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine"),
        default="constant",
    )
    train.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=DEFAULT_MINIMUM_LEARNING_RATE,
    )
    train.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    train.add_argument(
        "--minimum-face-confidence", type=float, default=MINIMUM_FACE_CONFIDENCE
    )
    train.add_argument(
        "--augmentation-strength",
        choices=tuple(AUGMENTATION_PROFILES),
        default="baseline",
    )
    train.add_argument(
        "--evaluation-interval", type=int, default=DEFAULT_EVALUATION_INTERVAL
    )
    train.add_argument("--pretrained-onnx", type=Path)
    train.add_argument(
        "--pretrained-trainable-scope",
        choices=("all", "fusion-head"),
        default="all",
    )
    train.add_argument(
        "--pretrained-loss",
        choices=("mean-angular", "tail-angular"),
        default="mean-angular",
    )
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--training-session-id", action="append", required=True)
    train.add_argument("--validation-session-id", action="append", required=True)
    train.set_defaults(action=run_train)

    freeze = subparsers.add_parser(
        "freeze-candidate",
        help="bind three candidate seeds and paired baselines into one frozen manifest",
    )
    freeze.add_argument("--data", type=Path, required=True)
    freeze.add_argument(
        "--candidate-report", type=Path, action="append", required=True
    )
    freeze.add_argument(
        "--baseline-report", type=Path, action="append", required=True
    )
    freeze.add_argument(
        "--changed-factor", choices=tuple(CANDIDATE_FACTOR_FIELDS), required=True
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(action=run_freeze_candidate)

    evaluate_command = subparsers.add_parser(
        "evaluate", help="evaluate a frozen checkpoint on held-out sessions"
    )
    evaluate_command.add_argument("--data", type=Path, required=True)
    evaluate_command.add_argument("--checkpoint", type=Path, required=True)
    evaluate_command.add_argument("--expected-checkpoint-sha256", required=True)
    evaluate_command.add_argument(
        "--expected-candidate-manifest-sha256", required=True
    )
    evaluate_command.add_argument("--output", type=Path, required=True)
    evaluate_command.add_argument("--batch-size", type=int, default=64)
    evaluate_command.add_argument("--pretrained-onnx", type=Path)
    evaluate_command.add_argument(
        "--validation-session-id", action="append", required=True
    )
    evaluate_command.set_defaults(action=run_evaluate)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.action(args))


if __name__ == "__main__":
    raise SystemExit(main())
