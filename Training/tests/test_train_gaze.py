import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_gaze import (
    REQUIRED_COLUMNS,
    angular_errors,
    gaze_vectors,
    load_samples,
    openvino_target_vectors,
    openvino_vectors_to_degrees,
)


class AngularMetricTests(unittest.TestCase):
    def test_identical_angles_have_zero_error(self):
        angles = np.array([[0.0, 0.0], [12.0, -7.0]])
        np.testing.assert_allclose(angular_errors(angles, angles), 0.0, atol=1e-6)

    def test_yaw_difference_matches_angular_error_at_zero_pitch(self):
        predicted = np.array([[10.0, 0.0]])
        expected = np.array([[-5.0, 0.0]])
        np.testing.assert_allclose(angular_errors(predicted, expected), 15.0, atol=1e-6)

    def test_gaze_vectors_are_unit_length(self):
        angles = np.array([[0.0, 0.0], [20.0, -15.0], [-8.0, 12.0]])
        lengths = np.linalg.norm(gaze_vectors(angles), axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-12)

    def test_openvino_vector_convention_round_trips_angles(self):
        angles = torch.tensor([[0.0, 0.0], [12.0, -7.0], [-8.0, 15.0]])
        actual = openvino_vectors_to_degrees(openvino_target_vectors(angles))
        torch.testing.assert_close(actual, angles, atol=1e-5, rtol=1e-5)


class DatasetLoadingTests(unittest.TestCase):
    def test_loads_only_a_complete_finished_session(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            left_name = "left.png"
            right_name = "right.png"
            Image.new("RGB", (60, 60)).save(session / left_name)
            Image.new("RGB", (60, 60)).save(session / right_name)
            metadata = {
                "schemaVersion": 2,
                "participantID": "participant",
                "sessionID": "session",
                "split": "training",
                "status": "finished",
                "capturedSamples": 100,
            }
            (session / "session.json").write_text(json.dumps(metadata))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 101):
                    row = {column: "0" for column in columns}
                    row.update(
                        {
                            "session_id": "session",
                            "split": "training",
                            "sample": str(sample_number),
                            "target_kind": "screen",
                            "left_image": left_name,
                            "right_image": right_name,
                        }
                    )
                    writer.writerow(row)

            samples, participants, sessions = load_samples(root, "training")

            self.assertEqual(len(samples), 100)
            self.assertEqual(participants, {"participant"})
            self.assertEqual(sessions, ["session"])

    def test_excludes_legacy_lens_samples_recorded_before_pose_settling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            metadata = {
                "schemaVersion": 1,
                "participantID": "participant",
                "sessionID": "session",
                "split": "training",
                "status": "finished",
                "capturedSamples": 100,
            }
            (session / "session.json").write_text(json.dumps(metadata))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 101):
                    row = {column: "0" for column in columns}
                    row.update(
                        {
                            "session_id": "session",
                            "split": "training",
                            "sample": str(sample_number),
                            "target_kind": "lens" if sample_number <= 6 else "screen",
                            "target_id": "0" if sample_number <= 6 else "1",
                            "left_image": "eye.png",
                            "right_image": "eye.png",
                        }
                    )
                    writer.writerow(row)

            samples, _, _ = load_samples(root, "training")

            self.assertEqual(len(samples), 94)


if __name__ == "__main__":
    unittest.main()
