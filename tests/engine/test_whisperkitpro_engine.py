# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Tests for WhisperKitProConfig.generate_cli_args.

The generated argument list must match what whisperkitpro-cli 3.x accepts:
the diarization backend is selected with `--diarizer` (there is no `--engine`
flag) and `--fast-load` does not exist. Released builds reject unknown
options, so nothing outside that surface may be emitted.
"""

import unittest
from pathlib import Path

from openbench.engine.whisperkitpro_engine import WhisperKitProConfig


MODEL_DIR = "/models/parakeet"


def _args(**overrides) -> list[str]:
    config = WhisperKitProConfig(model_dir=MODEL_DIR, **overrides)
    return config.generate_cli_args(model_path=Path(MODEL_DIR))


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


class TestCommonArgs(unittest.TestCase):
    def test_fast_load_is_never_emitted(self) -> None:
        for fast_load in (True, False):
            args = _args(fast_load=fast_load)
            self.assertNotIn("--fast-load", args)

    def test_no_diarization_flags_when_diarization_disabled(self) -> None:
        args = _args(diarization=False, engine="sortformer")
        for flag in ("--diarization", "--diarizer", "--engine", "--diarization-mode"):
            self.assertNotIn(flag, args)


class TestDiarizationArgs(unittest.TestCase):
    def test_pyannote_uses_diarizer_flag(self) -> None:
        args = _args(diarization=True, engine="pyannote")
        self.assertIn("--diarization", args)
        self.assertEqual(_value(args, "--diarizer"), "pyannote")
        self.assertNotIn("--engine", args)
        # Sortformer-only options must not leak into pyannote runs
        for flag in ("--diarization-mode", "--sortformer-model-version"):
            self.assertNotIn(flag, args)

    def test_sortformer_defaults_omit_optional_flags(self) -> None:
        args = _args(diarization=True, engine="sortformer")
        self.assertEqual(_value(args, "--diarizer"), "sortformer")
        self.assertEqual(_value(args, "--diarization-mode"), "prerecorded")
        for flag in ("--sortformer-model-version", "--sortformer-model-variant", "--speaker-models-path"):
            self.assertNotIn(flag, args)

    def test_sortformer_model_selection(self) -> None:
        args = _args(
            diarization=True,
            engine="sortformer",
            diarization_mode="realtime",
            sortformer_model_version="v3-preview",
            sortformer_model_variant="684_98MB",
            speaker_models_path="/speaker-models",
        )
        self.assertEqual(_value(args, "--diarization-mode"), "realtime")
        self.assertEqual(_value(args, "--sortformer-model-version"), "v3-preview")
        self.assertEqual(_value(args, "--sortformer-model-variant"), "684_98MB")
        self.assertEqual(_value(args, "--speaker-models-path"), "/speaker-models")

    def test_exclusive_reconciliation_flag(self) -> None:
        self.assertIn("--use-exclusive-reconciliation", _args(diarization=True, use_exclusive_reconciliation=True))
        self.assertNotIn("--use-exclusive-reconciliation", _args(diarization=True))


if __name__ == "__main__":
    unittest.main()
