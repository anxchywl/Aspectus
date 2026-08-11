import argparse
import contextlib
import csv
from dataclasses import replace
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_gaze
from train_gaze import (
    CANDIDATE_MANIFEST_FILENAME,
    CANONICAL_CROP_CONTRACT_V1,
    CANONICAL_CROP_CONTRACT_V2,
    CANONICAL_HEAD_POSE_CONTRACT,
    CANONICAL_INPUT_CONTRACT_V1,
    CANONICAL_INPUT_CONTRACT_V2,
    CANONICAL_LABEL_CONTRACT,
    CANONICAL_SCHEMA_CONTRACTS,
    CanonicalAlignmentEvidence,
    CHECKPOINT_FORMAT_VERSION,
    Evaluation,
    EyeQualityEvidence,
    FINAL_EVALUATION_FILENAME,
    GazeEstimator,
    LEGACY_INPUT_CONTRACT,
    NATIVE_ARCHITECTURE,
    NATIVE_PARAMETER_COUNT,
    POSE_PROMPTS,
    QualityGates,
    REQUIRED_COLUMNS,
    Sample,
    TARGETS_PER_POSE,
    aggregate_angular_training_loss,
    angular_cosine_loss,
    angular_errors,
    augmentation_config,
    build_candidate_manifest,
    canonical_columns,
    canonical_schema,
    claim_final_evaluation_session,
    epoch_evaluation,
    expected_target_label,
    file_digest,
    filtering_config,
    gaze_vectors,
    is_pretrained_fusion_head_parameter,
    input_preprocessing_contract,
    label_contract_digest,
    learning_rate_scheduler,
    load_checkpoint_model,
    load_samples,
    mirror_gaze_sample,
    model_digest,
    openvino_target_vectors,
    openvino_vectors_to_degrees,
    prepare_output_directory,
    protect_private_artifact,
    quality_gates,
    require_final_evaluation_output,
    require_frozen_candidate_manifest,
    require_final_session_created_after_freeze,
    require_development_gate_pass,
    require_single_recording_setup,
    require_single_new_completed_session,
    require_single_new_completed_session_hashes,
    require_stable_completed_session_ledger,
    require_untouched_final_session,
    restore_model_state,
    rgb_to_bgr,
    run_evaluate,
    sample_set_digest,
    canonical_quality_diagnostics,
    select_epoch,
    seed_everything,
    session_id_digest,
    snapshot_model_state,
    train_model,
    training_loss_configuration,
    validate_pose_coverage,
    validate_numeric_pitch_contract,
    validate_training_arguments,
    validated_completed_session_ledger,
    validated_filtering_config,
)


TEST_DISPLAY_GEOMETRY = {
    "displayWidthMM": 300.0,
    "displayHeightMM": 200.0,
    "viewingDistanceMM": 500.0,
}
TEST_VALIDATION_HEAD_POSES = (
    (0.0, 0.0, 0.0),
    (8.0, 0.0, 0.0),
    (-8.0, 0.0, 0.0),
    (0.0, -6.0, 0.0),
    (0.0, 6.0, 0.0),
)


def metadata(schema_version: int, split: str = "training") -> dict:
    value = {
        "schemaVersion": schema_version,
        "participantID": "participant",
        "sessionID": "session",
        "split": split,
        "status": "finished",
        "capturedSamples": 810,
        "samplesPerTarget": 6,
        "targetCount": 135,
        "displayGeometry": TEST_DISPLAY_GEOMETRY,
        "eyeImageWidth": 60,
        "eyeImageHeight": 60,
        "cameraFormat": "fixture camera 1280x720@30 BGRA",
        "createdAt": "2026-01-01T00:00:00Z",
        "completedAt": "2026-01-01T00:05:00Z",
    }
    if canonical_schema(schema_version):
        value.update({
            "sourceImageWidth": 1280,
            "sourceImageHeight": 720,
            "cropContract": CANONICAL_SCHEMA_CONTRACTS[schema_version][0],
            "labelContract": CANONICAL_LABEL_CONTRACT,
            "headPoseContract": CANONICAL_HEAD_POSE_CONTRACT,
        })
    return value


def manifest_row(columns: list[str], sample_number: int,
                 schema_version: int, split: str = "training") -> dict[str, str]:
    target_id = (sample_number - 1) // 6
    kind, target_x, target_y, target_yaw, target_pitch, pose = expected_target_label(
        split, target_id, TEST_DISPLAY_GEOMETRY
    )
    row = {column: "0" for column in columns}
    row.update({
        "schema_version": str(schema_version),
        "participant_id": "participant",
        "session_id": "session",
        "split": split,
        "sample": str(sample_number),
        "target_id": str(target_id),
        "target_kind": kind,
        "target_x": f"{target_x:.6f}",
        "target_y": f"{target_y:.6f}",
        "target_yaw_deg": f"{target_yaw:.6f}",
        "target_pitch_deg": f"{target_pitch:.6f}",
        "pose_prompt": pose,
        "head_yaw_deg": f"{TEST_VALIDATION_HEAD_POSES[target_id // 27][0]:.12f}",
        "head_pitch_deg": f"{TEST_VALIDATION_HEAD_POSES[target_id // 27][1]:.12f}",
        "head_roll_deg": f"{TEST_VALIDATION_HEAD_POSES[target_id // 27][2]:.12f}",
        "face_conf": "0.9",
        "open_l": "1",
        "open_r": "1",
        "left_image": "eye.png",
        "right_image": "eye.png",
    })
    if canonical_schema(schema_version):
        row.update({
            "frame_id": str(sample_number),
            "elapsed_s": f"{sample_number * 0.12:.12f}",
            "contour_points_l": "8",
            "contour_points_r": "8",
            "pupil_source_l": "contourCentroid",
            "pupil_source_r": "contourCentroid",
            "pupil_points_l": "0",
            "pupil_points_r": "0",
            "axis_start_x_l": "0.250000000000",
            "axis_start_y_l": "0.400000000000",
            "axis_end_x_l": "0.350000000000",
            "axis_end_y_l": "0.400000000000",
            "axis_start_x_r": "0.650000000000",
            "axis_start_y_r": "0.400000000000",
            "axis_end_x_r": "0.750000000000",
            "axis_end_y_r": "0.400000000000",
            "alignment_rotation_deg": "0.000000000000",
            "alignment_disagreement_deg": "0.000000000000",
            "crop_clipped_fraction_l": "0.000000000000",
            "crop_clipped_fraction_r": "0.000000000000",
        })
        # both fixture axes are 128 source px, so the shared v1 side and the per-eye v2 sides
        # agree at 1.8 x 128; asymmetric axes are exercised separately
        row.update(
            {"crop_side_px": "230.400000000000"}
            if schema_version == 4
            else {
                "crop_side_px_l": "230.400000000000",
                "crop_side_px_r": "230.400000000000",
            }
        )
    return row


def write_complete_session(root: Path, schema_version: int, split: str = "training",
                           row_mutator=None, metadata_mutator=None) -> Path:
    session = root / f"{split}-session"
    session.mkdir()
    Image.new("RGB", (60, 60)).save(session / "eye.png")
    session_metadata = metadata(schema_version, split)
    if metadata_mutator is not None:
        metadata_mutator(session_metadata)
    (session / "session.json").write_text(json.dumps(session_metadata))
    columns = sorted(
        REQUIRED_COLUMNS | (
            canonical_columns(schema_version) if canonical_schema(schema_version) else set()
        )
    )
    with (session / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sample_number in range(1, 811):
            row = manifest_row(columns, sample_number, schema_version, split)
            if row_mutator is not None:
                row_mutator(row, sample_number)
            writer.writerow(row)
    return session


def numeric_pitch_samples(
    session_id: str, eye_path: Path, invert_head_pitch: bool = False,
) -> list[Sample]:
    samples = []
    for target_id in range(len(POSE_PROMPTS) * TARGETS_PER_POSE):
        kind, target_x, target_y, yaw, pitch, pose = expected_target_label(
            "validation", target_id, TEST_DISPLAY_GEOMETRY
        )
        head_pose = TEST_VALIDATION_HEAD_POSES[target_id // TARGETS_PER_POSE]
        if invert_head_pitch:
            head_pose = (head_pose[0], -head_pose[1], head_pose[2])
        samples.append(Sample(
            session_id=session_id,
            split="validation",
            target_id=target_id,
            target_kind=kind,
            left_path=eye_path,
            right_path=eye_path,
            head_pose=head_pose,
            gaze_degrees=(yaw, pitch),
            pose_prompt=pose,
            target_position=(target_x, target_y),
            schema_version=3,
        ))
    return samples


class AngularMetricTests(unittest.TestCase):
    def test_python_target_plan_matches_fixed_swift_fixtures(self):
        lens = expected_target_label("training", 0, TEST_DISPLAY_GEOMETRY)
        training_centre = expected_target_label("training", 1, TEST_DISPLAY_GEOMETRY)
        validation_right = expected_target_label("validation", 1, TEST_DISPLAY_GEOMETRY)
        shifted_training = expected_target_label("training", 28, TEST_DISPLAY_GEOMETRY)

        self.assertEqual(lens, ("lens", 0.5, 0.0, 0.0, 0.0, "neutral"))
        self.assertEqual(training_centre[0:3], ("screen", 0.5, 0.5))
        self.assertAlmostEqual(training_centre[3], 0.0)
        self.assertAlmostEqual(training_centre[4], -11.309932474020213)
        self.assertEqual(validation_right[0:3], ("screen", 0.7, 0.5))
        self.assertAlmostEqual(validation_right[3], 6.84277341263094)
        self.assertEqual(shifted_training[0:3], ("screen", 0.1, 0.9))
        self.assertEqual(shifted_training[5], "turnLeft")

    def test_identical_angles_have_zero_error(self):
        angles = np.array([[0.0, 0.0], [12.0, -7.0]])
        np.testing.assert_allclose(angular_errors(angles, angles), 0.0, atol=1e-6)

    def test_yaw_difference_matches_angular_error_at_zero_pitch(self):
        predicted = np.array([[10.0, 0.0]])
        expected = np.array([[-5.0, 0.0]])
        np.testing.assert_allclose(angular_errors(predicted, expected), 15.0, atol=1e-6)

    def test_native_training_loss_is_angular_cosine_in_degree_output_space(self):
        expected = torch.tensor([[0.0, 0.0], [10.0, -5.0]])

        identical = angular_cosine_loss(expected, expected)
        displaced = angular_cosine_loss(expected + 10.0, expected)

        self.assertAlmostEqual(identical.item(), 0.0, places=7)
        self.assertGreater(displaced.item(), identical.item())

    def test_loss_provenance_names_the_angular_cosine_objective(self):
        expected = {
            "formulation": "angular-cosine",
            "tail_fraction": None,
            "tail_weight": None,
        }

        self.assertEqual(training_loss_configuration(False, "mean-angular"), expected)
        self.assertEqual(training_loss_configuration(True, "mean-angular"), expected)
        self.assertEqual(
            training_loss_configuration(True, "tail-angular"),
            {
                "formulation": "tail-angular-cosine",
                "tail_fraction": train_gaze.TAIL_LOSS_FRACTION,
                "tail_weight": train_gaze.TAIL_LOSS_WEIGHT,
            },
        )

    def test_native_compact_architecture_keeps_the_frozen_parameter_count(self):
        model = GazeEstimator()

        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            NATIVE_PARAMETER_COUNT,
        )
        self.assertEqual(NATIVE_PARAMETER_COUNT, 156_226)

    def test_gaze_vectors_are_unit_length(self):
        angles = np.array([[0.0, 0.0], [20.0, -15.0], [-8.0, 12.0]])
        lengths = np.linalg.norm(gaze_vectors(angles), axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-12)

    def test_openvino_vector_convention_round_trips_angles(self):
        angles = torch.tensor([[0.0, 0.0], [12.0, -7.0], [-8.0, 15.0]])
        actual = openvino_vectors_to_degrees(openvino_target_vectors(angles))
        torch.testing.assert_close(actual, angles, atol=1e-5, rtol=1e-5)

    def test_openvino_source_vector_uses_negative_z_at_neutral(self):
        raw = torch.tensor([
            [0.0, 0.0, -1.0],
            [-0.17364818, 0.0, -0.98480775],
            [0.0, -0.12186934, -0.99254615],
        ])
        expected = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, -7.0]])
        torch.testing.assert_close(
            openvino_vectors_to_degrees(raw), expected, atol=1e-4, rtol=1e-4
        )

    def test_openvino_eye_input_uses_bgr_channel_order(self):
        rgb = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
        expected = torch.tensor([[[[3.0]], [[2.0]], [[1.0]]]])
        torch.testing.assert_close(rgb_to_bgr(rgb), expected)

    def test_tail_loss_emphasises_the_worst_twenty_percent(self):
        losses = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

        mean = aggregate_angular_training_loss(losses, "mean-angular")
        tail = aggregate_angular_training_loss(losses, "tail-angular")

        self.assertAlmostEqual(mean.item(), 0.2)
        self.assertAlmostEqual(tail.item(), 0.4)

    def test_strong_augmentation_expands_every_continuous_range(self):
        baseline = augmentation_config("baseline")
        strong = augmentation_config("strong")

        for key in (
            "brightness_range", "contrast_range", "scale_range", "translation_pixels"
        ):
            self.assertLess(strong[key][0], baseline[key][0])
            self.assertGreater(strong[key][1], baseline[key][1])
        self.assertGreater(
            strong["head_pose_noise_standard_deviation_degrees"],
            baseline["head_pose_noise_standard_deviation_degrees"],
        )
        self.assertEqual(
            strong["paired_horizontal_mirror_probability"],
            baseline["paired_horizontal_mirror_probability"],
        )

    def test_mirroring_swaps_eyes_and_inverts_horizontal_geometry(self):
        left = Image.new("RGB", (2, 1))
        left.putpixel((0, 0), (1, 0, 0))
        right = Image.new("RGB", (2, 1))
        right.putpixel((1, 0), (2, 0, 0))

        mirrored_left, mirrored_right, head_pose, gaze = mirror_gaze_sample(
            left, right, (12.0, -4.0, 3.0), (8.0, -6.0))

        self.assertEqual(mirrored_left.getpixel((0, 0)), (2, 0, 0))
        self.assertEqual(mirrored_right.getpixel((1, 0)), (1, 0, 0))
        self.assertEqual(head_pose, (-12.0, -4.0, -3.0))
        self.assertEqual(gaze, (-8.0, -6.0))

class DatasetLoadingTests(unittest.TestCase):
    def test_schema_four_quality_diagnostics_report_raw_evidence_without_thresholds(self):
        left = EyeQualityEvidence(8, "visionLandmark", 1, (0.1, 0.2), (0.2, 0.2), 0.0, 120.0)
        right = EyeQualityEvidence(7, "contourCentroid", 0, (0.7, 0.2), (0.8, 0.2), 0.1, 96.0)
        alignment = CanonicalAlignmentEvidence(
            (1280, 720), left, right, 2.0, 4.0, 2,
        )
        sample = Sample(
            session_id="session", split="training", target_id=0, target_kind="lens",
            left_path=Path("left.png"), right_path=Path("right.png"),
            head_pose=(0, 0, 0), gaze_degrees=(0, 0), schema_version=5,
            canonical_alignment=alignment,
        )

        diagnostics = canonical_quality_diagnostics([sample])

        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertEqual(
            diagnostics["crop_clipped_fraction"],
            {"count": 2, "minimum": 0.0, "median": 0.05, "maximum": 0.1},
        )
        self.assertEqual(diagnostics["alignment_disagreement_degrees"]["median"], 4.0)
        self.assertEqual(
            diagnostics["crop_side_pixels"],
            {"count": 2, "minimum": 96.0, "median": 108.0, "maximum": 120.0},
        )
        self.assertEqual(diagnostics["crop_side_ratio"]["median"], 0.8)
        self.assertEqual(
            diagnostics["pupil_source_pair_counts"],
            {"visionLandmark+contourCentroid": 1},
        )

    def test_loads_only_a_complete_finished_session(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            (session / "session.json").write_text(json.dumps(metadata(2)))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 811):
                    writer.writerow(manifest_row(columns, sample_number, 2))

            samples, participants, sessions = load_samples(root, "training")

            self.assertEqual(len(samples), 810)
            self.assertEqual(participants, {"participant"})
            self.assertEqual(sessions, ["session"])
            self.assertEqual(samples[0].schema_version, 2)
            self.assertEqual(len(samples[0].participant_binding), 64)
            self.assertEqual(len(samples[0].setup_binding), 64)
            self.assertEqual(len(samples[0].session_metadata_sha256), 64)
            self.assertEqual(len(samples[0].manifest_sha256), 64)

    def test_canonical_schemas_load_only_their_frozen_contract(self):
        self.assertEqual(len(REQUIRED_COLUMNS | canonical_columns(4)), 41)
        self.assertEqual(len(REQUIRED_COLUMNS | canonical_columns(5)), 42)
        expected_contracts = {
            4: CANONICAL_INPUT_CONTRACT_V1,
            5: CANONICAL_INPUT_CONTRACT_V2,
        }
        for schema_version, expected_contract in expected_contracts.items():
            with self.subTest(schema_version=schema_version):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    write_complete_session(root, schema_version)

                    samples, _, _ = load_samples(root, "training")

                    self.assertEqual(len(samples), 810)
                    self.assertEqual(samples[0].input_contract, expected_contract)
                    self.assertEqual(
                        samples[0].label_contract_sha256, label_contract_digest()
                    )
                    evidence = samples[0].canonical_alignment
                    self.assertIsNotNone(evidence)
                    assert evidence is not None
                    self.assertEqual(evidence.source_image_size, (1280, 720))
                    self.assertEqual(evidence.left.crop_side_pixels, 230.4)
                    self.assertEqual(evidence.right.crop_side_pixels, 230.4)
                    changed = replace(
                        samples[0],
                        canonical_alignment=replace(
                            evidence,
                            left=replace(evidence.left, crop_side_pixels=230.5),
                        ),
                    )
                    self.assertNotEqual(
                        sample_set_digest([samples[0]]), sample_set_digest([changed])
                    )

    def test_v2_renders_each_eye_at_its_own_scale_while_v1_shares_the_longer(self):
        # the right axis is 0.05 wide against the left's 0.10, as head yaw foreshortens the far
        # eye. v1 scales both crops by 1.8 x the longer axis, so the far eye is rendered at half
        # scale; v2 gives each eye its own side, so both render identically.
        def narrow_right_eye(row, sample_number):
            row["axis_end_x_r"] = "0.700000000000"
            if "crop_side_px_r" in row:
                row["crop_side_px_r"] = "115.200000000000"

        def loaded_alignment(schema_version):
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                write_complete_session(
                    root, schema_version, row_mutator=narrow_right_eye
                )
                samples, _, _ = load_samples(root, "training")
                evidence = samples[0].canonical_alignment
                assert evidence is not None
                return evidence

        v1 = loaded_alignment(4)
        self.assertEqual(v1.left.crop_side_pixels, 230.4)
        self.assertEqual(v1.right.crop_side_pixels, 230.4)
        # the shared side comes from the near eye, so the far eye's own axis covers half as much
        # of its crop: rendered scale carries the head yaw that produced the foreshortening
        self.assertAlmostEqual(128.0 / v1.left.crop_side_pixels, 0.5555555555555556)
        self.assertAlmostEqual(64.0 / v1.right.crop_side_pixels, 0.2777777777777778)

        v2 = loaded_alignment(5)
        self.assertEqual(v2.left.crop_side_pixels, 230.4)
        self.assertEqual(v2.right.crop_side_pixels, 115.2)
        # each eye's own axis spans the same fraction of its own crop, so an eye's rendered scale
        # no longer depends on how far the head is turned away from it
        self.assertAlmostEqual(
            128.0 / v2.left.crop_side_pixels, 64.0 / v2.right.crop_side_pixels
        )
        self.assertAlmostEqual(64.0 / v2.right.crop_side_pixels, 1 / 1.8)

    def test_schema_four_rejects_missing_head_pose_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_complete_session(
                root,
                4,
                metadata_mutator=lambda value: value.pop("headPoseContract"),
            )

            with self.assertRaisesRegex(ValueError, "head-pose contract"):
                load_samples(root, "training")

    def test_canonical_schemas_reject_tampered_derived_alignment(self):
        # every serialized crop side must still be recomputable from the raw axes, including
        # each of v2's two per-eye sides
        tampered_columns = {
            4: ("alignment_rotation_deg", "crop_side_px"),
            5: ("alignment_rotation_deg", "crop_side_px_l", "crop_side_px_r"),
        }
        for schema_version, columns in tampered_columns.items():
            for column in columns:
                with self.subTest(schema_version=schema_version, column=column):
                    def tamper(row, sample_number, column=column):
                        if sample_number == 1:
                            row[column] = f"{float(row[column]) + 1:.12f}"

                    with tempfile.TemporaryDirectory() as temporary_directory:
                        root = Path(temporary_directory)
                        write_complete_session(root, schema_version, row_mutator=tamper)

                        with self.assertRaisesRegex(
                            ValueError, "derived alignment evidence"
                        ):
                            load_samples(root, "training")

    def test_schema_four_accepts_finite_out_of_frame_axes_with_recomputed_clipping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def cross_frame_boundary(row, sample_number):
                if sample_number == 1:
                    row["axis_start_x_l"] = "-0.050000000000"
                    row["axis_end_x_l"] = "0.050000000000"
                    row["crop_clipped_fraction_l"] = "0.500000000000"

            write_complete_session(root, 4, row_mutator=cross_frame_boundary)

            samples, _, _ = load_samples(root, "training")

            evidence = samples[0].canonical_alignment
            self.assertIsNotNone(evidence)
            assert evidence is not None
            self.assertEqual(evidence.left.axis_start[0], -0.05)
            self.assertEqual(evidence.left.crop_clipped_fraction, 0.5)

    def test_schema_four_rejects_tracking_quality_outside_unit_range(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def tamper(row, sample_number):
                if sample_number == 1:
                    row["open_l"] = "1.01"

            write_complete_session(root, 4, row_mutator=tamper)

            with self.assertRaisesRegex(ValueError, "tracking quality"):
                load_samples(root, "training")

    def test_schema_four_rejects_duplicate_headers_and_surplus_row_fields(self):
        for mutation in ("duplicate-header", "surplus-row-field"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    session = write_complete_session(root, 4)
                    manifest = session / "manifest.csv"
                    lines = manifest.read_text().splitlines()
                    if mutation == "duplicate-header":
                        lines[0] += ",schema_version"
                        lines[1] += ",4"
                    else:
                        lines[1] += ",unexpected"
                    manifest.write_text("\n".join(lines) + "\n")

                    with self.assertRaisesRegex(ValueError, "frozen contract"):
                        load_samples(root, "training")

    def test_schema_four_rejects_negative_and_nonincreasing_elapsed_time(self):
        for first, second, message in (
            ("-0.100000000000", None, "elapsed time is invalid"),
            ("0.120000000000", "0.120000000000", "not strictly increasing"),
        ):
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)

                    def tamper(row, sample_number):
                        if sample_number == 1:
                            row["elapsed_s"] = first
                        elif sample_number == 2 and second is not None:
                            row["elapsed_s"] = second

                    write_complete_session(root, 4, row_mutator=tamper)

                    with self.assertRaisesRegex(ValueError, message):
                        load_samples(root, "training")

    def test_excludes_legacy_lens_samples_recorded_before_pose_settling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            (session / "session.json").write_text(json.dumps(metadata(1)))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 811):
                    writer.writerow(manifest_row(columns, sample_number, 1))

            samples, _, _ = load_samples(root, "training")

            self.assertEqual(len(samples), 780)

    def test_excludes_rows_outside_the_tracking_quality_bounds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            (session / "session.json").write_text(json.dumps(metadata(3)))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 811):
                    row = manifest_row(columns, sample_number, 3)
                    if sample_number == 1:
                        row["head_roll_deg"] = "21"
                    if sample_number == 2:
                        row["face_conf"] = "0.69"
                    writer.writerow(row)

            samples, _, _ = load_samples(root, "training")

            self.assertEqual(len(samples), 808)

    def test_applies_the_configured_face_confidence_threshold(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            (session / "session.json").write_text(json.dumps(metadata(3)))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 811):
                    row = manifest_row(columns, sample_number, 3)
                    if sample_number == 1:
                        row["face_conf"] = "0.77"
                    writer.writerow(row)

            samples, _, _ = load_samples(
                root, "training", minimum_face_confidence=0.78
            )

            self.assertEqual(len(samples), 809)

    def test_validation_requires_every_pose_after_filtering(self):
        samples = [
            Sample(
                session_id="session", split="validation", target_id=pose * 27,
                target_kind="screen", left_path=Path("left.png"),
                right_path=Path("right.png"), head_pose=(0, 0, 0),
                gaze_degrees=(0, 0),
            )
            for pose in range(5)
            for _ in range(100 if pose != 4 else 99)
        ]

        with self.assertRaisesRegex(ValueError, "pose coverage is incomplete"):
            validate_pose_coverage(samples, ["session"])

    def test_validation_requires_representative_lens_coverage(self):
        samples = []
        for pose in range(5):
            for target_position, count, kind in (
                (0, 5, "lens"), (26, 5, "lens"), (1, 90, "screen")
            ):
                samples.extend(
                    Sample(
                        session_id="session", split="validation",
                        target_id=pose * TARGETS_PER_POSE + target_position,
                        target_kind=kind, left_path=Path("left.png"),
                        right_path=Path("right.png"),
                        head_pose=TEST_VALIDATION_HEAD_POSES[pose],
                        gaze_degrees=(0, 0),
                    )
                    for _ in range(count)
                )

        validate_pose_coverage(samples, ["session"])
        samples.pop(0)
        samples.append(
            Sample(
                session_id="session", split="validation", target_id=1,
                target_kind="screen", left_path=Path("left.png"),
                right_path=Path("right.png"), head_pose=(0, 0, 0),
                gaze_degrees=(0, 0),
            )
        )

        with self.assertRaisesRegex(ValueError, "lens coverage is incomplete"):
            validate_pose_coverage(samples, ["session"])

    def test_validation_requires_physical_head_pose_separation(self):
        samples = []
        for pose in range(5):
            for target_position, count, kind in (
                (0, 6, "lens"), (26, 6, "lens"), (1, 88, "screen")
            ):
                samples.extend(
                    Sample(
                        session_id="session", split="validation",
                        target_id=pose * TARGETS_PER_POSE + target_position,
                        target_kind=kind, left_path=Path("left.png"),
                        right_path=Path("right.png"), head_pose=(0, 0, 0),
                        gaze_degrees=(0, 0),
                    )
                    for _ in range(count)
                )

        with self.assertRaisesRegex(ValueError, "does not separate"):
            validate_pose_coverage(samples, ["session"])

    def test_numeric_pitch_contract_rejects_one_inverted_session(self):
        samples = numeric_pitch_samples("first", Path("eye.png"))
        samples.extend(numeric_pitch_samples(
            "second", Path("eye.png"), invert_head_pitch=True
        ))

        with self.assertRaisesRegex(ValueError, "negative for up and positive for down"):
            validate_numeric_pitch_contract(samples, ["first", "second"])

    def test_numeric_pitch_contract_rejects_nonmonotonic_screen_labels(self):
        samples = numeric_pitch_samples("session", Path("eye.png"))
        samples = [
            replace(sample, gaze_degrees=(sample.gaze_degrees[0], -0.1))
            if sample.target_kind == "screen" and sample.target_position[1] == 0.9
            else sample
            for sample in samples
        ]

        with self.assertRaisesRegex(ValueError, "not monotonic"):
            validate_numeric_pitch_contract(samples, ["session"])

    def test_validation_does_not_drop_a_fully_filtered_session(self):
        samples = [
            Sample(
                session_id="retained", split="validation", target_id=pose * 27,
                target_kind="screen", left_path=Path("left.png"),
                right_path=Path("right.png"), head_pose=(0, 0, 0),
                gaze_degrees=(0, 0),
            )
            for pose in range(5)
            for _ in range(100)
        ]

        with self.assertRaisesRegex(ValueError, "filtered-away"):
            validate_pose_coverage(samples, ["retained", "filtered-away"])

    def test_rejects_non_finite_tracking_quality(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            (session / "session.json").write_text(json.dumps(metadata(3)))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 811):
                    row = manifest_row(columns, sample_number, 3)
                    if sample_number == 1:
                        row["face_conf"] = "nan"
                    writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "non-finite face_conf"):
                load_samples(root, "training")

    def test_rejects_a_target_label_outside_the_recorded_plan(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            (session / "session.json").write_text(json.dumps(metadata(3)))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 811):
                    row = manifest_row(columns, sample_number, 3)
                    if sample_number == 7:
                        row["target_pitch_deg"] = "99"
                    writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "target plan mismatch"):
                load_samples(root, "training")

    def test_rejects_a_permuted_target_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            Image.new("RGB", (60, 60)).save(session / "eye.png")
            (session / "session.json").write_text(json.dumps(metadata(3)))
            columns = sorted(REQUIRED_COLUMNS)
            with (session / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for sample_number in range(1, 811):
                    row = manifest_row(columns, sample_number, 3)
                    if sample_number == 1:
                        row["target_id"] = "1"
                    writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "target order mismatch"):
                load_samples(root, "training")

    def test_rejects_a_truncated_finished_session(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / "training-session"
            session.mkdir()
            truncated = metadata(3)
            truncated["capturedSamples"] = 100
            (session / "session.json").write_text(json.dumps(truncated))
            (session / "manifest.csv").write_text(",".join(sorted(REQUIRED_COLUMNS)) + "\n")

            with self.assertRaisesRegex(ValueError, "not a complete collection"):
                load_samples(root, "training")

    def test_rejects_duplicate_finished_session_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for directory_name in ("first", "second"):
                session = root / directory_name
                session.mkdir()
                (session / "session.json").write_text(json.dumps(metadata(3)))

            with self.assertRaisesRegex(ValueError, "duplicate finished session id"):
                load_samples(root, "training")

    def test_sample_digest_binds_private_provenance_without_exposing_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            eye_path = Path(temporary_directory) / "eye.png"
            Image.new("RGB", (60, 60)).save(eye_path)
            common = {
                "session_id": "session", "split": "training", "target_id": 0,
                "target_kind": "lens", "left_path": eye_path, "right_path": eye_path,
                "head_pose": (0, 0, 0), "gaze_degrees": (0, 0),
                "schema_version": 3, "session_metadata_sha256": "metadata",
                "manifest_sha256": "manifest",
            }
            first = Sample(**common, participant_binding="first")
            second = Sample(**common, participant_binding="second")

            self.assertNotEqual(sample_set_digest([first]), sample_set_digest([second]))

    def test_schema_three_digest_keeps_the_legacy_record_shape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            eye_path = Path(temporary_directory) / "eye.png"
            Image.new("RGB", (60, 60)).save(eye_path)
            sample = Sample(
                session_id="session",
                split="training",
                target_id=0,
                target_kind="lens",
                left_path=eye_path,
                right_path=eye_path,
                head_pose=(1, 2, 3),
                gaze_degrees=(4, 5),
                pose_prompt="neutral",
                target_position=(0.5, 0),
                face_confidence=0.9,
                minimum_openness=0.8,
                schema_version=3,
                participant_binding="participant",
                setup_binding="setup",
                session_created_at="created",
                session_metadata_sha256="metadata",
                manifest_sha256="manifest",
                input_contract="must-not-enter-a-legacy-digest",
                label_contract_sha256="must-not-enter-a-legacy-digest",
            )
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
            expected = hashlib.sha256()
            expected.update(json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode())
            image_sha256 = file_digest(eye_path)
            for _ in range(2):
                expected.update(eye_path.name.encode())
                expected.update(image_sha256.encode())

            self.assertEqual(sample_set_digest([sample]), expected.hexdigest())


class CheckpointSelectionTests(unittest.TestCase):
    gates = QualityGates(2.0, 5.0, 3.0)

    @staticmethod
    def evaluation(median: float, p95: float, lens: float) -> Evaluation:
        return Evaluation(100, median, p95, 20, lens, 1.0, 1.0)

    def test_selection_uses_the_worst_session_instead_of_pooling(self):
        result = epoch_evaluation(5, 1e-4, {
            "strong": self.evaluation(1.0, 3.0, 1.5),
            "failing": self.evaluation(2.1, 4.0, 3.6),
        }, self.gates)

        self.assertAlmostEqual(result.selection_score, 1.2)
        self.assertAlmostEqual(result.worst_lens_ratio, 1.2)

    def test_selection_rejects_non_finite_session_metrics(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            epoch_evaluation(
                1, 1e-4,
                {"session": self.evaluation(float("nan"), 4.0, 2.0)},
                self.gates,
            )

    def test_selection_can_choose_an_epoch_before_the_final_one(self):
        earlier = epoch_evaluation(
            10, 1e-4, {"session": self.evaluation(1.5, 4.0, 2.4)}, self.gates
        )
        final = epoch_evaluation(
            20, 1e-4, {"session": self.evaluation(1.8, 4.8, 3.3)}, self.gates
        )

        self.assertEqual(select_epoch([earlier, final]).epoch, 10)

    def test_selection_ties_choose_the_earlier_epoch_deterministically(self):
        first = epoch_evaluation(
            5, 1e-4, {"session": self.evaluation(1.5, 4.0, 2.4)}, self.gates
        )
        later = epoch_evaluation(
            10, 1e-4, {"session": self.evaluation(1.5, 4.0, 2.4)}, self.gates
        )

        self.assertEqual(select_epoch([later, first]).epoch, 5)
        self.assertEqual(select_epoch([first, later]).epoch, 5)

    def test_restores_the_selected_model_state(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        selected = snapshot_model_state(model)
        with torch.no_grad():
            model.weight.fill_(2.0)

        restore_model_state(model, selected)

        self.assertEqual(model.weight.item(), 1.0)

    def test_fixed_seed_training_is_deterministic(self):
        def train_once() -> dict[str, torch.Tensor]:
            seed_everything(19)
            model = torch.nn.Linear(2, 1)
            optimiser = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
            features = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
            labels = torch.tensor([[0.2], [-0.1]])
            for _ in range(4):
                optimiser.zero_grad(set_to_none=True)
                loss = torch.nn.functional.smooth_l1_loss(model(features), labels)
                loss.backward()
                optimiser.step()
            return snapshot_model_state(model)

        first = train_once()
        second = train_once()

        for name in first:
            torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)

    def test_real_augmented_training_path_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            eye_path = Path(temporary_directory) / "eye.png"
            gradient = np.arange(60 * 60 * 3, dtype=np.uint8).reshape(60, 60, 3)
            Image.fromarray(gradient, mode="RGB").save(eye_path)

            def samples(session_id: str, count: int) -> list[Sample]:
                return [
                    Sample(
                        session_id=session_id,
                        split="training" if session_id == "training" else "validation",
                        target_id=index,
                        target_kind="lens" if index % 2 == 0 else "screen",
                        left_path=eye_path,
                        right_path=eye_path,
                        head_pose=(float(index), -float(index), float(index) / 2),
                        gaze_degrees=(float(index) / 2, -float(index) / 3),
                    )
                    for index in range(count)
                ]

            training = samples("training", 4)
            validation = samples("development-a", 2) + samples("development-b", 2)

            def train_once():
                with contextlib.redirect_stdout(io.StringIO()):
                    return train_model(
                        training,
                        validation,
                        ["development-a", "development-b"],
                        epochs=2,
                        batch_size=2,
                        learning_rate=1e-3,
                        learning_rate_schedule="constant",
                        minimum_learning_rate=1e-6,
                        augmentation_strength="baseline",
                        weight_decay=1e-4,
                        evaluation_interval=1,
                        seed=19,
                        gates=self.gates,
                    )

            first = train_once()
            second = train_once()

            self.assertEqual(first.selected_epoch, second.selected_epoch)
            self.assertEqual(first.evaluated_epochs, second.evaluated_epochs)
            self.assertEqual(model_digest(first.model), model_digest(second.model))

    def test_cosine_schedule_reaches_the_predeclared_minimum(self):
        model = torch.nn.Linear(1, 1)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = learning_rate_scheduler(optimiser, "cosine", 4, 1e-6)
        observed = [optimiser.param_groups[0]["lr"]]

        for _ in range(4):
            optimiser.step()
            scheduler.step()
            observed.append(optimiser.param_groups[0]["lr"])

        self.assertEqual(observed[0], 1e-4)
        self.assertAlmostEqual(observed[-1], 1e-6)
        self.assertTrue(all(left > right for left, right in zip(observed, observed[1:])))

    def test_fusion_head_scope_excludes_the_eye_feature_layers(self):
        self.assertTrue(is_pretrained_fusion_head_parameter(
            "model/tf/math/add_23/Add;model/conv2d_14/Conv2D.weight"
        ))
        self.assertTrue(is_pretrained_fusion_head_parameter(
            "model/tf/math/add_24/Add;model/conv2d_15/Conv2D.bias"
        ))
        self.assertTrue(is_pretrained_fusion_head_parameter("Identity.weight"))
        self.assertFalse(is_pretrained_fusion_head_parameter(
            "model/tf/math/add_22/Add;model/conv2d_13/Conv2D.weight"
        ))


class ArtifactSafetyTests(unittest.TestCase):
    @staticmethod
    def training_arguments(**overrides):
        values = {
            "epochs": 1,
            "batch_size": 1,
            "evaluation_interval": 1,
            "weight_decay": 1e-4,
            "minimum_face_confidence": 0.70,
            "learning_rate": 1e-4,
            "minimum_learning_rate": 1e-6,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_final_session_must_not_have_been_used_for_development(self):
        with self.assertRaisesRegex(ValueError, "already consumed"):
            require_untouched_final_session(
                ["development"], ["training"], ["development"]
            )

    def test_final_session_must_be_the_only_completion_after_freezing(self):
        require_single_new_completed_session(
            ["final"], ["training", "development"],
            ["training", "development", "final"],
        )

        with self.assertRaisesRegex(ValueError, "exactly the one completed session"):
            require_single_new_completed_session(
                ["final"], ["training"], ["training", "diagnostic", "final"]
            )

    def test_hashed_completed_session_snapshot_preserves_the_same_boundary(self):
        known = [session_id_digest("training"), session_id_digest("development")]
        require_single_new_completed_session_hashes(
            ["final"], known, ["training", "development", "final"]
        )

        with self.assertRaisesRegex(ValueError, "exactly the one completed session"):
            require_single_new_completed_session_hashes(
                ["final"], known,
                ["training", "development", "diagnostic", "final"],
            )

    def test_final_session_must_begin_after_the_candidate_freeze(self):
        common = {
            "session_id": "final", "split": "validation", "target_id": 0,
            "target_kind": "lens", "left_path": Path("left.png"),
            "right_path": Path("right.png"), "head_pose": (0, 0, 0),
            "gaze_degrees": (0, 0),
        }
        require_final_session_created_after_freeze(
            [Sample(
                **common, session_created_at="2026-01-02T00:00:01+00:00"
            )],
            "2026-01-02T00:00:00+00:00",
        )

        with self.assertRaisesRegex(ValueError, "before the candidate was frozen"):
            require_final_session_created_after_freeze(
                [Sample(
                    **common, session_created_at="2026-01-02T00:00:00+00:00"
                )],
                "2026-01-02T00:00:00+00:00",
            )

    def test_final_session_claim_is_private_global_and_one_time(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_session_id = "private-final-session"
            with patch.object(
                train_gaze,
                "durably_sync_published_path",
                wraps=train_gaze.durably_sync_published_path,
            ) as durable_sync:
                claim = claim_final_evaluation_session(
                    root, raw_session_id, "c" * 64, "a" * 64
                )

            ledger_path = root / train_gaze.FINAL_EVALUATION_LEDGER_FILENAME
            lock_path = root / train_gaze.FINAL_EVALUATION_LEDGER_LOCK_FILENAME
            durable_sync.assert_called_once_with(ledger_path)
            self.assertEqual(claim, session_id_digest(raw_session_id))
            self.assertNotIn(raw_session_id, ledger_path.read_text())
            self.assertEqual(stat.S_IMODE(ledger_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(ValueError, "already been evaluated"):
                claim_final_evaluation_session(
                    root, raw_session_id, "d" * 64, "b" * 64
                )
            with self.assertRaisesRegex(ValueError, "candidate has already used"):
                claim_final_evaluation_session(
                    root, "another-final-session", "c" * 64, "a" * 64
                )
            claim_final_evaluation_session(
                root, "another-final-session", "d" * 64, "b" * 64
            )

    def test_final_output_path_is_fixed_next_to_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "GazeEstimator.pt"
            required = root / FINAL_EVALUATION_FILENAME

            require_final_evaluation_output(checkpoint, required)
            with self.assertRaisesRegex(ValueError, "output must be"):
                require_final_evaluation_output(checkpoint, root / "another.json")

    def test_completed_session_ledger_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "completed-session ledger"):
            validated_completed_session_ledger(["session", "session"])

    def test_training_aborts_if_the_completed_session_ledger_changes(self):
        self.assertEqual(
            require_stable_completed_session_ledger(
                ["development", "training"], ["training", "development"]
            ),
            ["development", "training"],
        )
        with self.assertRaisesRegex(ValueError, "changed during training"):
            require_stable_completed_session_ledger(
                ["training", "development"],
                ["training", "development", "concurrent"],
            )

    def test_training_rejects_non_finite_hyperparameters(self):
        for name in ("weight_decay", "learning_rate", "minimum_learning_rate"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "finite"):
                validate_training_arguments(
                    self.training_arguments(**{name: float("nan")})
                )

    def test_personalized_sessions_require_one_recording_setup(self):
        common = {
            "session_id": "session", "split": "training", "target_id": 0,
            "target_kind": "lens", "left_path": Path("left.png"),
            "right_path": Path("right.png"), "head_pose": (0, 0, 0),
            "gaze_degrees": (0, 0),
        }
        self.assertEqual(
            require_single_recording_setup(
                [Sample(**common, setup_binding="same")],
                [Sample(**common, setup_binding="same")],
            ),
            "same",
        )
        with self.assertRaisesRegex(ValueError, "one recording setup"):
            require_single_recording_setup(
                [Sample(**common, setup_binding="first")],
                [Sample(**common, setup_binding="second")],
            )

    def test_final_evaluation_requires_every_development_session_to_pass(self):
        passing = Evaluation(100, 1.0, 4.0, 20, 2.0, 1.0, 1.0)
        failing = Evaluation(100, 2.1, 4.0, 20, 2.0, 1.0, 1.0)
        require_development_gate_pass(
            {"first": passing, "second": passing},
            ["first", "second"], quality_gates(),
        )
        with self.assertRaisesRegex(ValueError, "every development session"):
            require_development_gate_pass(
                {"first": passing, "second": failing},
                ["first", "second"], quality_gates(),
            )

    def test_nonempty_output_directory_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "evaluation.json").write_text("immutable")

            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare_output_directory(output)

    def test_model_development_artifacts_are_owner_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "run"
            prepare_output_directory(output)
            artifact = output / "checkpoint"
            artifact.write_text("private")
            protect_private_artifact(artifact)

            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(artifact.stat().st_mode), 0o600)

    def test_json_is_published_atomically_after_the_final_safety_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "report.json"
            checked = []

            def verify_before_publication():
                self.assertFalse(output.exists())
                checked.append(True)

            with patch.object(
                train_gaze,
                "durably_sync_published_path",
                wraps=train_gaze.durably_sync_published_path,
            ) as durable_sync:
                train_gaze.write_json_exclusive(
                    output,
                    {"passed": True},
                    before_publish=verify_before_publication,
                )

            self.assertEqual(checked, [True])
            durable_sync.assert_called_once_with(output)
            self.assertEqual(json.loads(output.read_text()), {"passed": True})
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

            rejected = root / "rejected.json"

            def reject_publication():
                raise RuntimeError("snapshot changed")

            with self.assertRaisesRegex(RuntimeError, "changed"):
                train_gaze.write_json_exclusive(
                    rejected,
                    {"passed": False},
                    before_publish=reject_publication,
                )
            self.assertFalse(rejected.exists())

    def test_durable_publication_syncs_the_file_and_parent_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "report.json"
            output.write_text("durable")
            synced_types = []

            def record_file_type(descriptor):
                mode = os.fstat(descriptor).st_mode
                synced_types.append(
                    "directory" if stat.S_ISDIR(mode) else "file"
                )

            with patch.object(
                train_gaze,
                "sync_to_permanent_storage",
                side_effect=record_file_type,
            ):
                train_gaze.durably_sync_published_path(output)

            self.assertEqual(synced_types, ["file", "directory"])

    def test_durable_sync_uses_fullfsync_when_available(self):
        with patch.object(
            train_gaze.fcntl, "F_FULLFSYNC", 51, create=True
        ), patch.object(train_gaze.fcntl, "fcntl") as full_sync:
            train_gaze.sync_to_permanent_storage(17)

        full_sync.assert_called_once_with(17, 51)

    def test_frozen_filtering_accepts_a_recorded_face_threshold(self):
        expected = filtering_config(0.78)

        self.assertEqual(validated_filtering_config(expected), expected)

    def test_frozen_filtering_rejects_unrecorded_filter_changes(self):
        filtering = filtering_config(0.78)
        filtering["minimum_openness"] = 0.5

        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validated_filtering_config(filtering)

    def test_frozen_filtering_rejects_a_threshold_below_the_recorder_floor(self):
        with self.assertRaisesRegex(ValueError, "face-confidence filter is invalid"):
            validated_filtering_config(filtering_config(0.69))

    def test_quality_gates_are_canonical_and_not_configurable(self):
        self.assertEqual(quality_gates(), QualityGates(2.0, 5.0, 3.0))


class CandidateManifestTests(unittest.TestCase):
    @staticmethod
    def write_run(
        root: Path, name: str, seed: int, score: float,
        augmentation_strength: str, weight_decay: float = 1e-4,
        training_data_sha256: str = "1" * 64,
        development_data_sha256: str = "3" * 64,
        source_training_data_sha256: str = "4" * 64,
        minimum_face_confidence: float = 0.70,
        training_samples: int = 810,
        development_samples: int = 1_000,
        input_contract: str = LEGACY_INPUT_CONTRACT,
    ) -> Path:
        output = root / name
        output.mkdir()
        evaluation = Evaluation(
            count=500,
            angular_median_degrees=score * 2,
            angular_p95_degrees=score * 5,
            lens_count=60,
            lens_p95_degrees=score * 3,
            yaw_mae_degrees=1,
            pitch_mae_degrees=1,
        )
        session_evaluations = {
            "private-development-a": evaluation.__dict__,
            "private-development-b": evaluation.__dict__,
        }
        selection_components = (
            train_gaze.selection_components(
                {
                    "private-development-a": evaluation,
                    "private-development-b": evaluation,
                },
                quality_gates(),
            )
            if np.isfinite(score) else (score, score, score, score)
        )
        selected_score = selection_components[0]
        selection = {
            "epoch": 40,
            "learning_rate": 1e-4,
            "selection_score": selected_score,
            "worst_lens_ratio": selection_components[1],
            "worst_p95_ratio": selection_components[2],
            "worst_median_ratio": selection_components[3],
            "session_evaluations": session_evaluations,
        }
        passed = all(
            train_gaze.gate_passes(value, quality_gates())
            for value in (evaluation, evaluation)
        )
        model = GazeEstimator().eval()
        model_sha256 = model_digest(model)
        optimizer = {
            "name": "AdamW", "learning_rate": 1e-4,
            "weight_decay": weight_decay, "scheduler": "constant",
            "minimum_learning_rate": None, "scheduler_step_unit": None,
        }
        fine_tuning = {
            "scope": "all", "trainable_parameters": NATIVE_PARAMETER_COUNT,
            "total_parameters": NATIVE_PARAMETER_COUNT,
            "trainable_parameter_names": ["weight"],
        }
        shared = {
            "architecture": NATIVE_ARCHITECTURE,
            "pretrained_model_sha256": None,
            "training_sessions": ["private-training"],
            "development_sessions": [
                "private-development-a", "private-development-b"
            ],
            "known_completed_sessions": [
                "private-training", "private-development-a", "private-development-b"
            ],
            "frozen_at": "2026-01-01T00:00:00+00:00",
            "setup_sha256": "2" * 64,
            "training_data_sha256": training_data_sha256,
            "development_data_sha256": development_data_sha256,
            "source_training_data_sha256": source_training_data_sha256,
            "source_development_data_sha256": "5" * 64,
            "training_samples": training_samples,
            "development_samples": development_samples,
            "source_training_samples": 810,
            "source_development_samples": 1_000,
            "training_epochs": 80,
            "evaluation_interval": 1,
            "selected_epoch": 40,
            "seed": seed,
            "batch_size": 64,
            "fine_tuning": fine_tuning,
            "loss": training_loss_configuration(False, "mean-angular"),
            "optimizer": optimizer,
            "filtering": filtering_config(minimum_face_confidence),
            "augmentation": augmentation_config(augmentation_strength),
            "input_preprocessing": input_preprocessing_contract(input_contract),
            "label_contract_sha256": label_contract_digest(),
            "software_versions": train_gaze.software_versions(
                train_gaze.execution_device()
            ),
            "trainer_sha256": file_digest(Path(train_gaze.__file__).resolve()),
        }
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state": model.state_dict(),
            "model_sha256": model_sha256,
            "validation_sessions": shared["development_sessions"],
            "trainable_parameters": NATIVE_PARAMETER_COUNT,
            "selection": selection,
            "evaluated_epochs": [selection],
            "quality_gates": quality_gates().__dict__,
            "development_passed": passed,
            "development_session_evaluations": session_evaluations,
            **shared,
        }
        checkpoint_path = output / "GazeEstimator.pt"
        torch.save(checkpoint, checkpoint_path)
        report = {
            "schema_version": CHECKPOINT_FORMAT_VERSION,
            "output_contract": "gaze_degrees",
            "checkpoint_sha256": file_digest(checkpoint_path),
            "model_sha256": model_sha256,
            "trainable_parameters": NATIVE_PARAMETER_COUNT,
            "session_evaluations": session_evaluations,
            "selection": selection,
            "evaluated_epochs": [selection],
            "gates": {**quality_gates().__dict__, "passed": passed},
            **shared,
        }
        report_path = output / "evaluation.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report_path

    def cohorts(self, root: Path, candidate_scores=(0.70, 0.72, 0.74)):
        seeds = (7, 19, 43)
        candidate = [
            self.write_run(
                root, f"candidate-{seed}", seed, score, "strong"
            )
            for seed, score in zip(seeds, candidate_scores)
        ]
        baseline = [
            self.write_run(
                root, f"baseline-{seed}", seed, score, "baseline"
            )
            for seed, score in zip(seeds, (0.80, 0.82, 0.84))
        ]
        return candidate, baseline

    @staticmethod
    def completed_hashes():
        return [
            session_id_digest(value) for value in (
                "private-training", "private-development-a", "private-development-b"
            )
        ]

    def test_manifest_binds_three_paired_seeds_and_the_seed_seven_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root)
            output = candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME)
            manifest = build_candidate_manifest(
                candidate, baseline, "augmentation", self.completed_hashes(),
                output, "2026-01-02T00:00:00+00:00",
            )
            output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            checkpoint = candidate[0].with_name("GazeEstimator.pt")

            actual = require_frozen_candidate_manifest(
                checkpoint, file_digest(checkpoint), file_digest(output)
            )

            self.assertEqual(manifest["schema_version"], 2)
            self.assertTrue(actual["robustness"]["passed"])
            self.assertEqual(actual["primary_seed"], 7)
            self.assertNotIn("private-training", output.read_text())
            wrong_checkpoint = candidate[1].with_name("GazeEstimator.pt")
            with self.assertRaisesRegex(ValueError, "manifest is missing"):
                require_frozen_candidate_manifest(
                    wrong_checkpoint,
                    file_digest(wrong_checkpoint),
                    file_digest(output),
                )

    def test_manifest_treats_input_preprocessing_as_one_major_factor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seeds = (7, 19, 43)
            candidate = [
                self.write_run(
                    root,
                    f"candidate-{seed}",
                    seed,
                    score,
                    "baseline",
                    input_contract=CANONICAL_INPUT_CONTRACT_V2,
                )
                for seed, score in zip(seeds, (0.70, 0.72, 0.74))
            ]
            baseline = [
                self.write_run(
                    root,
                    f"baseline-{seed}",
                    seed,
                    score,
                    "baseline",
                    input_contract=LEGACY_INPUT_CONTRACT,
                )
                for seed, score in zip(seeds, (0.80, 0.82, 0.84))
            ]

            manifest = build_candidate_manifest(
                candidate,
                baseline,
                "input",
                self.completed_hashes(),
                candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                "2026-01-02T00:00:00+00:00",
            )

            self.assertEqual(manifest["changed_factor"], "input")

    def test_candidate_report_output_contract_must_match_the_checkpoint_architecture(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = self.write_run(
                Path(temporary_directory), "candidate", 7, 0.70, "baseline"
            )
            report = json.loads(report_path.read_text())
            report["output_contract"] = "gaze_vector"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

            with self.assertRaisesRegex(ValueError, "output contract"):
                train_gaze.validated_seed_run(report_path)

    def test_candidate_identity_ignores_paths_timestamps_and_final_attempts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root)
            first = build_candidate_manifest(
                candidate, baseline, "augmentation", self.completed_hashes(),
                candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                "2026-01-02T00:00:00+00:00",
            )
            second = build_candidate_manifest(
                candidate, baseline, "augmentation",
                [*self.completed_hashes(), session_id_digest("failed-final")],
                root / "copied-output" / CANDIDATE_MANIFEST_FILENAME,
                "2026-01-03T00:00:00+00:00",
            )

            self.assertNotEqual(first["frozen_at"], second["frozen_at"])
            self.assertNotEqual(
                first["known_completed_session_sha256"],
                second["known_completed_session_sha256"],
            )
            self.assertEqual(
                first["candidate_identity_sha256"],
                second["candidate_identity_sha256"],
            )

    def test_candidate_identity_ignores_a_replacement_baseline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seeds = (7, 19, 43)
            candidate = [
                self.write_run(
                    root, f"candidate-{seed}", seed, score, "baseline",
                    weight_decay=1e-3,
                )
                for seed, score in zip(seeds, (0.70, 0.72, 0.74))
            ]
            first_baseline = [
                self.write_run(
                    root, f"first-baseline-{seed}", seed, score, "baseline",
                    weight_decay=1e-4,
                )
                for seed, score in zip(seeds, (0.80, 0.82, 0.84))
            ]
            replacement_baseline = [
                self.write_run(
                    root, f"replacement-baseline-{seed}", seed, score,
                    "baseline", weight_decay=2e-4,
                )
                for seed, score in zip(seeds, (0.90, 0.92, 0.94))
            ]

            first = build_candidate_manifest(
                candidate, first_baseline, "optimizer", self.completed_hashes(),
                root / "first" / CANDIDATE_MANIFEST_FILENAME,
                "2026-01-02T00:00:00+00:00",
            )
            second = build_candidate_manifest(
                candidate, replacement_baseline, "optimizer",
                self.completed_hashes(),
                root / "second" / CANDIDATE_MANIFEST_FILENAME,
                "2026-01-03T00:00:00+00:00",
            )

            self.assertNotEqual(
                first["baseline_procedure_sha256"],
                second["baseline_procedure_sha256"],
            )
            self.assertNotEqual(first["paired_scores"], second["paired_scores"])
            self.assertEqual(
                first["candidate_identity_sha256"],
                second["candidate_identity_sha256"],
            )

    def test_manifest_rejects_a_missing_or_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root)
            duplicate = self.write_run(
                root, "candidate-duplicate", 19, 0.73, "strong"
            )

            with self.assertRaisesRegex(ValueError, "seeds"):
                build_candidate_manifest(
                    [candidate[0], candidate[1], duplicate], baseline,
                    "augmentation", self.completed_hashes(),
                    candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                    "2026-01-02T00:00:00+00:00",
                )

    def test_manifest_rejects_a_failing_candidate_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root, (0.70, 1.01, 0.74))

            with self.assertRaisesRegex(ValueError, "every candidate seed"):
                build_candidate_manifest(
                    candidate, baseline, "augmentation", self.completed_hashes(),
                    candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                    "2026-01-02T00:00:00+00:00",
                )

    def test_manifest_rejects_a_non_improving_seed_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root, (0.70, 0.83, 0.74))

            with self.assertRaisesRegex(ValueError, "not consistent"):
                build_candidate_manifest(
                    candidate, baseline, "augmentation", self.completed_hashes(),
                    candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                    "2026-01-02T00:00:00+00:00",
                )

    def test_manifest_rejects_procedure_drift_across_candidate_seeds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root)
            candidate[2] = self.write_run(
                root, "candidate-43-drift", 43, 0.74, "strong", weight_decay=1e-3
            )

            with self.assertRaisesRegex(ValueError, "procedure differs across seeds"):
                build_candidate_manifest(
                    candidate, baseline, "augmentation", self.completed_hashes(),
                    candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                    "2026-01-02T00:00:00+00:00",
                )

    def test_manifest_rejects_an_undeclared_second_factor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seeds = (7, 19, 43)
            candidate = [
                self.write_run(
                    root, f"candidate-{seed}", seed, score, "strong",
                    weight_decay=1e-3,
                )
                for seed, score in zip(seeds, (0.70, 0.72, 0.74))
            ]
            baseline = [
                self.write_run(
                    root, f"baseline-{seed}", seed, score, "baseline"
                )
                for seed, score in zip(seeds, (0.80, 0.82, 0.84))
            ]

            with self.assertRaisesRegex(ValueError, "outside the declared"):
                build_candidate_manifest(
                    candidate, baseline, "augmentation", self.completed_hashes(),
                    candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                    "2026-01-02T00:00:00+00:00",
                )

    def test_manifest_allows_filtering_to_change_only_retained_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seeds = (7, 19, 43)
            candidate = [
                self.write_run(
                    root, f"filtered-candidate-{seed}", seed, score, "baseline",
                    training_data_sha256="6" * 64,
                    development_data_sha256="7" * 64,
                    minimum_face_confidence=0.78,
                    training_samples=760,
                    development_samples=940,
                )
                for seed, score in zip(seeds, (0.70, 0.72, 0.74))
            ]
            baseline = [
                self.write_run(
                    root, f"unfiltered-baseline-{seed}", seed, score, "baseline"
                )
                for seed, score in zip(seeds, (0.80, 0.82, 0.84))
            ]

            manifest = build_candidate_manifest(
                candidate, baseline, "filtering", self.completed_hashes(),
                candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                "2026-01-02T00:00:00+00:00",
            )

            self.assertTrue(manifest["robustness"]["passed"])
            self.assertEqual(manifest["changed_factor"], "filtering")

    def test_manifest_rejects_a_different_data_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, _ = self.cohorts(root)
            baseline = [
                self.write_run(
                    root, f"different-data-baseline-{seed}", seed, score,
                    "baseline", source_training_data_sha256="9" * 64,
                )
                for seed, score in zip((7, 19, 43), (0.80, 0.82, 0.84))
            ]

            with self.assertRaisesRegex(ValueError, "data or environments differ"):
                build_candidate_manifest(
                    candidate, baseline, "augmentation", self.completed_hashes(),
                    candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                    "2026-01-02T00:00:00+00:00",
                )

    def test_manifest_rejects_a_non_finite_seed_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root)
            candidate[2] = self.write_run(
                root, "candidate-43-nonfinite", 43, float("nan"), "strong"
            )

            with self.assertRaisesRegex(ValueError, "evaluations are invalid"):
                build_candidate_manifest(
                    candidate, baseline, "augmentation", self.completed_hashes(),
                    candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME),
                    "2026-01-02T00:00:00+00:00",
                )

    def test_manifest_rejects_an_altered_report_after_freezing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root)
            output = candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME)
            manifest = build_candidate_manifest(
                candidate, baseline, "augmentation", self.completed_hashes(),
                output, "2026-01-02T00:00:00+00:00",
            )
            output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            expected_manifest_sha256 = file_digest(output)
            candidate[1].write_text(candidate[1].read_text() + "\n")
            checkpoint = candidate[0].with_name("GazeEstimator.pt")

            with self.assertRaisesRegex(ValueError, "does not match its artifacts"):
                require_frozen_candidate_manifest(
                    checkpoint, file_digest(checkpoint), expected_manifest_sha256
                )

    def test_manifest_requires_its_predeclared_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate, baseline = self.cohorts(root)
            output = candidate[0].with_name(CANDIDATE_MANIFEST_FILENAME)
            manifest = build_candidate_manifest(
                candidate, baseline, "augmentation", self.completed_hashes(),
                output, "2026-01-02T00:00:00+00:00",
            )
            output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            checkpoint = candidate[0].with_name("GazeEstimator.pt")

            with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
                require_frozen_candidate_manifest(
                    checkpoint, file_digest(checkpoint), "0" * 64
                )


class CheckpointLoadingTests(unittest.TestCase):
    @staticmethod
    def covered_samples(
        session_id: str, eye_path: Path,
        created_at: str = "2026-01-01T00:00:00+00:00",
    ) -> list[Sample]:
        samples = []
        for pose in range(5):
            target_counts = [(0, 6), (26, 6)] + [
                (position, 4 if position <= 13 else 3)
                for position in range(1, 26)
            ]
            for target_position, count in target_counts:
                target_id = pose * TARGETS_PER_POSE + target_position
                kind, target_x, target_y, yaw, pitch, prompt = expected_target_label(
                    "validation", target_id, TEST_DISPLAY_GEOMETRY
                )
                samples.extend(
                    Sample(
                        session_id=session_id,
                        split="validation",
                        target_id=target_id,
                        target_kind=kind,
                        left_path=eye_path,
                        right_path=eye_path,
                        head_pose=TEST_VALIDATION_HEAD_POSES[pose],
                        gaze_degrees=(yaw, pitch),
                        pose_prompt=prompt,
                        target_position=(target_x, target_y),
                        schema_version=3,
                        participant_binding="participant-binding",
                        setup_binding="setup-binding",
                        session_created_at=created_at,
                        session_metadata_sha256=f"{session_id}-metadata",
                        manifest_sha256=f"{session_id}-manifest",
                    )
                    for _ in range(count)
                )
        return samples

    def test_version_four_evaluator_writes_one_immutable_final_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            eye_path = root / "eye.png"
            Image.new("RGB", (60, 60)).save(eye_path)
            training = self.covered_samples("training", eye_path)
            development_a = self.covered_samples("development-a", eye_path)
            development_b = self.covered_samples("development-b", eye_path)
            development = development_a + development_b
            final = self.covered_samples(
                "final", eye_path, created_at="2026-01-03T00:00:00+00:00"
            )
            model = GazeEstimator().eval()
            seed_everything(7)
            checkpoint_payload = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "architecture": NATIVE_ARCHITECTURE,
                "model_state": model.state_dict(),
                "model_sha256": model_digest(model),
                "pretrained_model_sha256": None,
                "training_sessions": ["training"],
                "development_sessions": ["development-a", "development-b"],
                "known_completed_sessions": [
                    "training", "development-a", "development-b"
                ],
                "frozen_at": "2026-01-02T00:00:00+00:00",
                "setup_sha256": "setup-binding",
                "training_data_sha256": sample_set_digest(training),
                "development_data_sha256": sample_set_digest(development),
                "source_training_data_sha256": sample_set_digest(training),
                "source_development_data_sha256": sample_set_digest(development),
                "source_training_samples": len(training),
                "source_development_samples": len(development),
                "quality_gates": quality_gates().__dict__,
                "development_passed": True,
                "development_session_evaluations": {
                    "development-a": Evaluation(
                        500, 1.0, 2.0, 60, 2.0, 1.0, 1.0
                    ).__dict__,
                    "development-b": Evaluation(
                        500, 1.0, 2.0, 60, 2.0, 1.0, 1.0
                    ).__dict__,
                },
                "filtering": filtering_config(),
                "software_versions": train_gaze.software_versions(
                    train_gaze.execution_device()
                ),
                "trainer_sha256": file_digest(Path(train_gaze.__file__).resolve()),
                "seed": 7,
                "batch_size": 32,
                "optimizer": {},
                "fine_tuning": {},
                "loss": {},
                "augmentation": augmentation_config("baseline"),
                "input_preprocessing": input_preprocessing_contract(
                    LEGACY_INPUT_CONTRACT
                ),
                "label_contract_sha256": label_contract_digest(),
                "selected_epoch": 1,
                "selection": {},
                "evaluated_epochs": [],
            }
            checkpoint_path = root / "GazeEstimator.pt"
            torch.save(checkpoint_payload, checkpoint_path)
            candidate_manifest_path = root / CANDIDATE_MANIFEST_FILENAME
            candidate_manifest_path.write_text("{}\n")
            output_path = root / FINAL_EVALUATION_FILENAME
            args = argparse.Namespace(
                data=root,
                checkpoint=checkpoint_path,
                expected_checkpoint_sha256=file_digest(checkpoint_path),
                expected_candidate_manifest_sha256=file_digest(
                    candidate_manifest_path
                ),
                output=output_path,
                pretrained_onnx=None,
                validation_session_id=["final"],
                batch_size=32,
            )
            passing = Evaluation(500, 1.0, 2.0, 60, 2.0, 1.0, 1.0)

            def loaded_samples(_root, _split, selected, **_kwargs):
                if selected == {"development-a", "development-b"}:
                    return development, {"participant"}, [
                        "development-a", "development-b"
                    ]
                session_id = next(iter(selected))
                values = {"training": training, "final": final}[session_id]
                return values, {"participant"}, [session_id]

            def evaluated_sessions(_model, _samples, sessions, *_args):
                return {session_id: passing for session_id in sessions}

            candidate_manifest = {
                "frozen_at": "2026-01-02T00:00:00+00:00",
                "candidate_identity_sha256": "c" * 64,
                "known_completed_session_sha256": [
                    session_id_digest(value) for value in (
                        "training", "development-a", "development-b"
                    )
                ],
                "robustness": {"passed": True},
            }
            ledger_root = root / "global-ledger"
            with patch("train_gaze.finished_session_ids", return_value=[
                "training", "development-a", "development-b", "final"
            ]), patch("train_gaze.load_samples", side_effect=loaded_samples), patch(
                "train_gaze.evaluate_sessions", side_effect=evaluated_sessions
            ), patch("train_gaze.evaluate_samples", return_value=passing), patch(
                "train_gaze.diagnostic_breakdowns", return_value={}
            ), patch(
                "train_gaze.require_frozen_candidate_manifest",
                return_value=candidate_manifest,
            ), patch(
                "train_gaze.final_evaluation_ledger_root", return_value=ledger_root
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run_evaluate(args), 0)

            report = json.loads(output_path.read_text())
            self.assertTrue(report["gates"]["passed"])
            self.assertEqual(report["validation_sessions"], ["final"])
            self.assertEqual(
                report["input_preprocessing"],
                input_preprocessing_contract(LEGACY_INPUT_CONTRACT),
            )
            self.assertEqual(report["label_contract_sha256"], label_contract_digest())
            ledger = json.loads(
                (ledger_root / train_gaze.FINAL_EVALUATION_LEDGER_FILENAME).read_text()
            )
            self.assertNotIn("final", json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_evaluate(args)

    def test_final_evaluator_rejects_a_version_two_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            checkpoint_path = output_directory / "GazeEstimator.pt"
            model = GazeEstimator().eval()
            torch.save(
                {
                    "format_version": 2,
                    "architecture": NATIVE_ARCHITECTURE,
                    "model_state": model.state_dict(),
                    "model_sha256": model_digest(model),
                },
                checkpoint_path,
            )
            args = argparse.Namespace(
                checkpoint=checkpoint_path,
                expected_checkpoint_sha256=file_digest(checkpoint_path),
                output=output_directory / FINAL_EVALUATION_FILENAME,
                pretrained_onnx=None,
            )

            with self.assertRaisesRegex(ValueError, "version-4 frozen checkpoint"):
                run_evaluate(args)

    def test_loads_a_frozen_native_checkpoint_without_changing_its_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = GazeEstimator().eval()
            digest = model_digest(model)
            for format_version in (1, 2, 3):
                with self.subTest(format_version=format_version):
                    checkpoint_path = (
                        Path(temporary_directory) / f"GazeEstimator-{format_version}.pt"
                    )
                    torch.save(
                        {
                            "format_version": format_version,
                            "architecture": NATIVE_ARCHITECTURE,
                            "model_state": model.state_dict(),
                            "model_sha256": digest,
                            "training_sessions": ["training-session"],
                        },
                        checkpoint_path,
                    )

                    loaded, checkpoint, pretrained = load_checkpoint_model(
                        checkpoint_path, None
                    )

                    self.assertFalse(pretrained)
                    self.assertEqual(
                        checkpoint["training_sessions"], ["training-session"]
                    )
                    self.assertEqual(model_digest(loaded), digest)

    def test_rejects_a_checkpoint_with_a_mismatched_model_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "GazeEstimator.pt"
            model = GazeEstimator().eval()
            torch.save(
                {
                    "format_version": 1,
                    "architecture": NATIVE_ARCHITECTURE,
                    "model_state": model.state_dict(),
                    "model_sha256": "incorrect",
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "checkpoint model checksum mismatch"):
                load_checkpoint_model(checkpoint_path, None)


if __name__ == "__main__":
    unittest.main()
