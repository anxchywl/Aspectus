#!/usr/bin/env python3
"""train and convert the offline Aspectus appearance-based gaze estimator"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import coremltools as ct
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 60
HEAD_POSE_SCALE_DEGREES = 30.0
REQUIRED_COLUMNS = {
    "session_id",
    "split",
    "sample",
    "target_kind",
    "target_yaw_deg",
    "target_pitch_deg",
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
    "left_image",
    "right_image",
}


@dataclass(frozen=True)
class Sample:
    session_id: str
    split: str
    target_kind: str
    left_path: Path
    right_path: Path
    head_pose: tuple[float, float, float]
    gaze_degrees: tuple[float, float]


@dataclass(frozen=True)
class Evaluation:
    count: int
    angular_median_degrees: float
    angular_p95_degrees: float
    lens_count: int
    lens_p95_degrees: float
    yaw_mae_degrees: float
    pitch_mae_degrees: float


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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        return self.features(eye)


class GazeEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eye_encoder = EyeEncoder()
        self.regressor = nn.Sequential(
            nn.Linear(64 * 2 + 3, 128),
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


class GazeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]):
    def __init__(self, samples: list[Sample], augment: bool) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        left = self._load_eye(sample.left_path)
        right = self._load_eye(sample.right_path)
        if self.augment:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            left = ImageEnhance.Contrast(ImageEnhance.Brightness(left).enhance(brightness)).enhance(contrast)
            right = ImageEnhance.Contrast(ImageEnhance.Brightness(right).enhance(brightness)).enhance(contrast)
        return (
            image_tensor(left),
            image_tensor(right),
            torch.tensor(sample.head_pose, dtype=torch.float32),
            torch.tensor(sample.gaze_degrees, dtype=torch.float32),
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


def load_samples(root: Path, split: str) -> tuple[list[Sample], set[str], list[str]]:
    samples: list[Sample] = []
    participants: set[str] = set()
    session_ids: list[str] = []
    metadata_files = sorted(root.rglob("session.json"))
    if not metadata_files:
        raise ValueError(f"no session.json files found under {root}")

    for metadata_path in metadata_files:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("split") != split or metadata.get("status") != "finished":
            continue
        session_id = str(metadata["sessionID"])
        participants.add(str(metadata["participantID"]))
        session_ids.append(session_id)
        manifest_path = metadata_path.with_name("manifest.csv")
        with manifest_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
            rows = list(reader)
        expected = int(metadata["capturedSamples"])
        if expected != len(rows):
            raise ValueError(
                f"{session_id} metadata records {expected} samples but manifest has {len(rows)}"
            )
        if len(rows) < 100:
            raise ValueError(f"{session_id} is too small to be a complete collection")
        for row in rows:
            if row["session_id"] != session_id or row["split"] != split:
                raise ValueError(f"manifest identity mismatch in {manifest_path}")
            left_path = metadata_path.parent / row["left_image"]
            right_path = metadata_path.parent / row["right_image"]
            if not left_path.is_file() or not right_path.is_file():
                raise ValueError(f"missing eye image for sample {row['sample']} in {session_id}")
            samples.append(
                Sample(
                    session_id=session_id,
                    split=split,
                    target_kind=row["target_kind"],
                    left_path=left_path,
                    right_path=right_path,
                    head_pose=(
                        float(row["head_yaw_deg"]),
                        float(row["head_pitch_deg"]),
                        float(row["head_roll_deg"]),
                    ),
                    gaze_degrees=(
                        float(row["target_yaw_deg"]),
                        float(row["target_pitch_deg"]),
                    ),
                )
            )
    if not samples:
        raise ValueError(f"no finished {split} sessions found under {root}")
    return samples, participants, session_ids


def gaze_vectors(angles: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(angles)
    yaw = radians[:, 0]
    pitch = radians[:, 1]
    return np.stack(
        (np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)),
        axis=1,
    )


def angular_errors(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    products = np.sum(gaze_vectors(predicted) * gaze_vectors(expected), axis=1)
    return np.rad2deg(np.arccos(np.clip(products, -1.0, 1.0)))


def percentile(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(values, fraction * 100, method="linear"))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Evaluation:
    predictions: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    lens_masks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for left, right, pose, labels, is_lens in loader:
            result = model(left.to(device), right.to(device), pose.to(device))
            predictions.append(result.cpu().numpy())
            expected.append(labels.numpy())
            lens_masks.append(is_lens.numpy().astype(bool))
    predicted_array = np.concatenate(predictions)
    expected_array = np.concatenate(expected)
    lens_mask = np.concatenate(lens_masks)
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_model(
    training: list[Sample],
    validation: list[Sample],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[GazeEstimator, Evaluation, int]:
    seed_everything(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        GazeDataset(training, augment=True),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        GazeDataset(validation, augment=False), batch_size=batch_size, shuffle=False
    )
    model = GazeEstimator().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for left, right, pose, labels, _ in training_loader:
            optimiser.zero_grad(set_to_none=True)
            output = model(left.to(device), right.to(device), pose.to(device))
            loss = loss_function(output, labels.to(device))
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch {epoch:03d} loss={statistics.fmean(losses):.4f}")

    final_evaluation = evaluate(model, validation_loader, device)
    model.cpu().eval()
    return model, final_evaluation, epochs


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


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def gate_passes(evaluation: Evaluation, args: argparse.Namespace) -> bool:
    return (
        evaluation.angular_median_degrees <= args.max_median
        and evaluation.angular_p95_degrees <= args.max_p95
        and evaluation.lens_p95_degrees <= args.max_lens_p95
    )


def run_train(args: argparse.Namespace) -> int:
    training, training_participants, training_sessions = load_samples(args.data, "training")
    validation, validation_participants, validation_sessions = load_samples(args.data, "validation")
    participants = training_participants | validation_participants
    if len(participants) != 1 or training_participants != validation_participants:
        raise ValueError("the personalized gate requires training and validation from one participant")
    if set(training_sessions) & set(validation_sessions):
        raise ValueError("training and validation sessions overlap")

    args.output.mkdir(parents=True, exist_ok=True)
    model, evaluation, training_epochs = train_model(
        training,
        validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    checkpoint = args.output / "GazeEstimator.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_sha256": model_digest(model),
            "training_sessions": training_sessions,
            "validation_sessions": validation_sessions,
            "training_epochs": training_epochs,
        },
        checkpoint,
    )
    passed = gate_passes(evaluation, args)
    report = {
        "schema_version": 1,
        "architecture": "shared-eye-cnn-head-pose-v1",
        "model_sha256": model_digest(model),
        "seed": args.seed,
        "training_epochs": training_epochs,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "training_sessions": training_sessions,
        "validation_sessions": validation_sessions,
        "evaluation": asdict(evaluation),
        "gates": {
            "maximum_median_degrees": args.max_median,
            "maximum_p95_degrees": args.max_p95,
            "maximum_lens_p95_degrees": args.max_lens_p95,
            "passed": passed,
        },
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))
    if not passed:
        print("quality gate failed; checkpoint retained but Core ML export withheld", file=sys.stderr)
        return 2
    model_path = args.output / "GazeEstimator.mlpackage"
    export_coreml(model, model_path)
    print(f"quality gate passed; exported {model_path}")
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    model = GazeEstimator().eval()
    with tempfile.TemporaryDirectory(prefix="aspectus-gaze-smoke-") as directory:
        destination = Path(directory) / "GazeEstimator.mlpackage"
        export_coreml(model, destination)
        specification = ct.utils.load_spec(str(destination))
        inputs = [feature.name for feature in specification.description.input]
        outputs = [feature.name for feature in specification.description.output]
        if inputs != ["left_eye", "right_eye", "head_pose"] or outputs != ["gaze_degrees"]:
            raise RuntimeError(f"unexpected Core ML contract: inputs={inputs}, outputs={outputs}")
        generator = np.random.default_rng(args.seed)
        left = Image.fromarray(generator.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        right = Image.fromarray(generator.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        head_pose = np.array([[8.0, -5.0, 2.0]], dtype=np.float32)
        with torch.inference_mode():
            expected = model(
                image_tensor(left).unsqueeze(0),
                image_tensor(right).unsqueeze(0),
                torch.from_numpy(head_pose),
            ).numpy()
        runtime = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_ONLY)
        actual = np.asarray(
            runtime.predict(
                {"left_eye": left, "right_eye": right, "head_pose": head_pose}
            )["gaze_degrees"]
        )
        maximum_difference = float(np.max(np.abs(actual - expected)))
        if maximum_difference > 0.05:
            raise RuntimeError(
                f"Core ML output differs from PyTorch by {maximum_difference:.4f} degrees"
            )
        print(
            f"conversion smoke passed: {sum(parameter.numel() for parameter in model.parameters())} "
            f"parameters, max difference={maximum_difference:.4f}deg, "
            f"inputs={inputs}, outputs={outputs}"
        )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="convert an untrained model and verify its contract")
    smoke.add_argument("--seed", type=int, default=7)
    smoke.set_defaults(action=run_smoke)

    train = subparsers.add_parser("train", help="train on complete sessions and evaluate held out")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--max-median", type=float, default=2.0)
    train.add_argument("--max-p95", type=float, default=5.0)
    train.add_argument("--max-lens-p95", type=float, default=3.0)
    train.set_defaults(action=run_train)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.action(args))


if __name__ == "__main__":
    raise SystemExit(main())
