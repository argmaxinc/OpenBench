# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from openbench.dataset.dataset_base import DatasetConfig
from openbench.dataset.dataset_diarization import DiarizationDataset
from openbench.dataset.dataset_orchestration import OrchestrationDataset
from openbench.dataset.dataset_streaming_transcription import StreamingDataset
from openbench.dataset.dataset_transcription import TranscriptionDataset
from openbench.dataset.local_dataset_loader import load_local_dataset


def create_test_audio_file(audio_path: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> None:
    """Create a test audio file."""
    num_samples = int(duration_seconds * sample_rate)
    waveform = np.random.randn(num_samples).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate)


def create_local_dataset(
    dataset_dir: Path,
    audio_ids: list[str],
    reference_data_fn: Callable[[str], dict],  # Function that takes audio_id and returns reference data dict
    metadata: dict[str, dict] | None = None,
    split: str = "test",
) -> None:
    """Create a local dataset structure.

    Args:
        dataset_dir: Path to dataset directory
        audio_ids: List of audio IDs (without extension)
        reference_data_fn: Function that takes audio_id and returns reference data dict
        metadata: Optional metadata dict mapping audio_id to extra_info
        split: Split name (default: "test")
    """
    audio_dir = dataset_dir / "audio"
    reference_dir = dataset_dir / "reference"
    splits_dir = dataset_dir / "splits"

    # Create directories
    audio_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    # Create audio files and reference JSON files
    for audio_id in audio_ids:
        create_test_audio_file(audio_dir / f"{audio_id}.wav")

        # Get reference data for this audio_id
        reference_data = reference_data_fn(audio_id)
        with open(reference_dir / f"{audio_id}.json", "w") as f:
            json.dump(reference_data, f)

    # Create split file
    with open(splits_dir / f"{split}.txt", "w") as f:
        for audio_id in audio_ids:
            f.write(f"{audio_id}\n")

    # Create metadata file if provided
    if metadata:
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)


class TestLocalDatasetLoader(unittest.TestCase):
    """Test the local dataset loader functionality."""

    def test_load_local_dataset_diarization(self) -> None:
        """Test loading a local diarization dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "test_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {
                    "timestamps_start": [0.0, 5.2],
                    "timestamps_end": [5.2, 10.5],
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                }

            create_local_dataset(
                dataset_dir, audio_ids=["sample_001", "sample_002"], reference_data_fn=reference_data_fn, split="test"
            )

            # Load dataset
            dataset = load_local_dataset(dataset_dir, split="test")

            # Verify dataset
            self.assertEqual(len(dataset), 2)
            self.assertIn("audio", dataset.column_names)
            self.assertIn("timestamps_start", dataset.column_names)
            self.assertIn("timestamps_end", dataset.column_names)
            self.assertIn("speakers", dataset.column_names)

            # Verify first sample
            sample = dataset[0]
            self.assertIn("audio", sample)
            self.assertIn("timestamps_start", sample)
            self.assertEqual(len(sample["timestamps_start"]), 2)

    def test_load_local_dataset_with_metadata(self) -> None:
        """Test loading a local dataset with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "test_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {"transcript": ["hello", "world"]}

            metadata = {"sample_001": {"language": "en", "dictionary": ["hello", "world"]}}

            create_local_dataset(
                dataset_dir,
                audio_ids=["sample_001"],
                reference_data_fn=reference_data_fn,
                metadata=metadata,
                split="test",
            )

            # Load dataset
            dataset = load_local_dataset(dataset_dir, split="test")

            # Verify dataset
            self.assertEqual(len(dataset), 1)


class TestDiarizationDatasetLocal(unittest.TestCase):
    """Test DiarizationDataset with local datasets."""

    def test_diarization_dataset_from_local_path(self) -> None:
        """Test DiarizationDataset loading from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "diarization_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {
                    "timestamps_start": [0.0, 5.2],
                    "timestamps_end": [5.2, 10.5],
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                }

            create_local_dataset(
                dataset_dir, audio_ids=["sample_001"], reference_data_fn=reference_data_fn, split="test"
            )

            # Create config with local path
            config = DatasetConfig(dataset_id=str(dataset_dir), split="test")

            # Load dataset
            dataset = DiarizationDataset.from_config(config)

            # Verify dataset
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.dataset_name, "diarization_dataset")
            self.assertEqual(dataset.split, "test")
            self.assertEqual(dataset.organization, "local")

            # Verify sample
            sample = dataset[0]
            self.assertIsNotNone(sample.reference)
            self.assertEqual(len(sample.reference.timestamps_start), 2)
            self.assertEqual(len(sample.reference.speakers), 2)

    def test_diarization_dataset_with_uem(self) -> None:
        """Test DiarizationDataset with UEM timestamps in metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "diarization_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {
                    "timestamps_start": [0.0, 5.2],
                    "timestamps_end": [5.2, 10.5],
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                }

            metadata = {
                "sample_001": {
                    "uem_timestamps": [[0.0, 10.5]],
                }
            }

            create_local_dataset(
                dataset_dir,
                audio_ids=["sample_001"],
                reference_data_fn=reference_data_fn,
                metadata=metadata,
                split="test",
            )

            config = DatasetConfig(dataset_id=str(dataset_dir), split="test")
            dataset = DiarizationDataset.from_config(config)

            sample = dataset[0]
            self.assertIsNotNone(sample.uem)


class TestTranscriptionDatasetLocal(unittest.TestCase):
    """Test TranscriptionDataset with local datasets."""

    def test_transcription_dataset_from_local_path(self) -> None:
        """Test TranscriptionDataset loading from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "transcription_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {
                    "transcript": ["hello", "world", "how", "are", "you"],
                    "word_timestamps_start": [0.0, 0.5, 1.0, 1.5, 2.0],
                    "word_timestamps_end": [0.5, 1.0, 1.5, 2.0, 2.5],
                }

            metadata = {
                "sample_001": {
                    "language": "en",
                    "dictionary": ["hello", "world"],
                }
            }

            create_local_dataset(
                dataset_dir,
                audio_ids=["sample_001"],
                reference_data_fn=reference_data_fn,
                metadata=metadata,
                split="test",
            )

            config = DatasetConfig(dataset_id=str(dataset_dir), split="test")
            dataset = TranscriptionDataset.from_config(config)

            # Verify dataset
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.dataset_name, "transcription_dataset")

            # Verify sample
            sample = dataset[0]
            self.assertIsNotNone(sample.reference)
            self.assertEqual(len(sample.reference.words), 5)
            self.assertEqual(sample.language, "en")
            self.assertEqual(sample.dictionary, ["hello", "world"])


class TestStreamingDatasetLocal(unittest.TestCase):
    """Test StreamingDataset with local datasets."""

    def test_streaming_dataset_from_local_path(self) -> None:
        """Test StreamingDataset loading from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "streaming_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {
                    "text": "hello world how are you",
                    "word_detail": [
                        {"start": 0.0, "stop": 8000.0},
                        {"start": 8000.0, "stop": 16000.0},
                        {"start": 16000.0, "stop": 24000.0},
                        {"start": 24000.0, "stop": 32000.0},
                        {"start": 32000.0, "stop": 40000.0},
                    ],
                }

            create_local_dataset(
                dataset_dir, audio_ids=["sample_001"], reference_data_fn=reference_data_fn, split="test"
            )

            config = DatasetConfig(dataset_id=str(dataset_dir), split="test")
            dataset = StreamingDataset.from_config(config)

            # Verify dataset
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.dataset_name, "streaming_dataset")

            # Verify sample
            sample = dataset[0]
            self.assertIsNotNone(sample.reference)
            self.assertEqual(len(sample.reference.words), 5)

    def test_streaming_dataset_without_word_detail(self) -> None:
        """Test StreamingDataset without word_detail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "streaming_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {"text": "hello world"}

            create_local_dataset(
                dataset_dir, audio_ids=["sample_001"], reference_data_fn=reference_data_fn, split="test"
            )

            config = DatasetConfig(dataset_id=str(dataset_dir), split="test")
            dataset = StreamingDataset.from_config(config)

            sample = dataset[0]
            self.assertIsNotNone(sample.reference)
            self.assertEqual(len(sample.reference.words), 2)


class TestOrchestrationDatasetLocal(unittest.TestCase):
    """Test OrchestrationDataset with local datasets."""

    def test_orchestration_dataset_from_local_path(self) -> None:
        """Test OrchestrationDataset loading from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "orchestration_dataset"

            def reference_data_fn(audio_id: str) -> dict:
                return {
                    "transcript": ["hello", "world", "how", "are", "you"],
                    "word_speakers": ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01"],
                    "word_timestamps_start": [0.0, 0.5, 1.0, 1.5, 2.0],
                    "word_timestamps_end": [0.5, 1.0, 1.5, 2.0, 2.5],
                }

            create_local_dataset(
                dataset_dir, audio_ids=["sample_001"], reference_data_fn=reference_data_fn, split="test"
            )

            config = DatasetConfig(dataset_id=str(dataset_dir), split="test")
            dataset = OrchestrationDataset.from_config(config)

            # Verify dataset
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.dataset_name, "orchestration_dataset")

            # Verify sample
            sample = dataset[0]
            self.assertIsNotNone(sample.reference)
            self.assertEqual(len(sample.reference.words), 5)
            self.assertTrue(sample.reference.has_speakers)
            self.assertIsNotNone(sample.reference.get_speakers())
