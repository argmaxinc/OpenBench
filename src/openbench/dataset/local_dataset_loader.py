# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
from pathlib import Path

from argmaxtools.utils import get_logger
from datasets import Audio, Dataset, DatasetInfo, NamedSplit


logger = get_logger(__name__)

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".m4a"]


def load_json(json_file: Path) -> dict:
    """Load a JSON file.

    Args:
        file: Path to JSON file

    Returns:
        Dictionary with JSON data
    """
    with json_file.open("r") as f:
        return json.load(f)


def validate_local_dataset_structure(dataset_dir: Path, split: str) -> None:
    """Validate that the local dataset has the expected structure.

    Args:
        dataset_dir: Path to local dataset directory
        split: Split name (e.g., "test", "train")

    Raises:
        ValueError: If the dataset structure is invalid
    """
    # Check required directories exist
    audio_dir = dataset_dir / "audio"
    reference_dir = dataset_dir / "reference"
    splits_dir = dataset_dir / "splits"

    if not audio_dir.exists() or not audio_dir.is_dir():
        raise ValueError(f"Audio directory not found: {audio_dir}")

    if not reference_dir.exists() or not reference_dir.is_dir():
        raise ValueError(f"Reference directory not found: {reference_dir}")

    if not splits_dir.exists() or not splits_dir.is_dir():
        raise ValueError(f"Splits directory not found: {splits_dir}")

    # Check split file exists
    split_file = splits_dir / f"{split}.txt"
    if not split_file.exists():
        raise ValueError(f"Split file not found: {split_file}")


def parse_split_file(split_file: Path) -> list[str]:
    """Parse split file to get list of audio IDs.

    Args:
        split_file: Path to split file (e.g., test.txt)

    Returns:
        List of audio IDs (without extension)
    """
    audio_ids = []
    with open(split_file, "r") as f:
        for line in f:
            audio_id = line.strip()
            if audio_id:  # Skip empty lines
                audio_ids.append(audio_id)
    return audio_ids


def load_local_dataset(dataset_dir: Path, split: str) -> Dataset:
    """Load a local dataset and return a HuggingFace Dataset object.

    Args:
        dataset_dir: Path to local dataset directory
        split: Split name (e.g., "test", "train")

    Returns:
        HuggingFace Dataset with Audio column cast
    """
    # Validate structure
    validate_local_dataset_structure(dataset_dir, split)

    # Get paths
    audio_dir = dataset_dir / "audio"
    reference_dir = dataset_dir / "reference"
    splits_dir = dataset_dir / "splits"
    metadata_file = dataset_dir / "metadata.json"

    # Parse split file
    audio_ids = parse_split_file(splits_dir / f"{split}.txt")

    if not audio_ids:
        raise ValueError(f"No audio IDs found in split file: {splits_dir / f'{split}.txt'}")

    # Load metadata (optional)
    metadata = load_json(metadata_file) if metadata_file.exists() else {}

    # Build dataset rows
    rows = []
    for audio_id in audio_ids:
        # Find audio file (try common extensions)
        audio_file = None
        for ext in AUDIO_EXTENSIONS:
            candidate = audio_dir / f"{audio_id}{ext}"
            if candidate.exists():
                audio_file = candidate
                break

        if audio_file is None:
            raise ValueError(f"Audio file not found for audio_id: {audio_id} in {audio_dir}")

        # Load reference file
        reference_file = reference_dir / f"{audio_id}.json"
        if not reference_file.exists():
            raise ValueError(f"Reference file not found for audio_id: {audio_id} in {reference_dir}")

        reference_data = load_json(reference_file)

        # Build row: audio path + reference data + optional metadata
        row = {
            "audio": str(audio_file),
            **reference_data,
        }

        # Add metadata if available
        if audio_id in metadata:
            # Metadata goes into extra_info, but we'll handle that in prepare_sample
            # For now, we can add it as additional columns if needed
            # But typically metadata is handled separately
            pass

        rows.append(row)

    # Create HuggingFace Dataset with info set
    dataset = Dataset.from_list(
        rows, info=DatasetInfo(dataset_name=dataset_dir.name, config_name=None), split=NamedSplit(split)
    )

    # Cast audio column to Audio type
    dataset = dataset.cast_column("audio", Audio())

    return dataset
