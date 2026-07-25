# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, TypeVar

import numpy as np
import soundfile as sf
from argmaxtools.utils import get_logger
from datasets import Dataset as HfDataset
from datasets import load_dataset
from pydantic import BaseModel, Field

from ..types import PredictionProtocol
from .dataset_utils import validate_hf_dataset_schema


logger = get_logger(__name__)

ReferenceType = TypeVar("ReferenceType", bound=PredictionProtocol)  # For the reference/ground_truth object
ExtraInfoType = TypeVar("ExtraInfoType", bound=dict[str, Any])
SampleType = TypeVar("SampleType", bound="BaseSample")


class DatasetConfig(BaseModel):
    """Configuration for any dataset type."""

    dataset_id: str = Field(..., description="HuggingFace dataset ID. Ex: 'talkbank/callhome'")
    subset: str | None = Field(None, description="Subset of the dataset. Ex. 'eng' for the English subset.")
    split: str | None = Field(None, description="Split of the dataset")
    num_samples: int | None = Field(
        None, description="Number of samples to take from the dataset. If None, take all samples."
    )
    num_shards: int | None = Field(
        None,
        description=(
            "Split the dataset into this many shards and keep only `shard_index`, so one sweep "
            "can be spread over several machines or CI jobs. Shards are interleaved (row i goes "
            "to shard i % num_shards) rather than contiguous, so per-sample cost — e.g. voice-clone "
            "reference length — spreads evenly instead of piling into one shard."
        ),
    )
    shard_index: int = Field(0, description="Which shard to keep when `num_shards` is set (0-based).")
    exclude_sample_ids: frozenset[str] | None = Field(
        None,
        description=(
            "Drop rows whose `sample_id_column` value appears in this set. Lets an interrupted "
            "sweep resume without recomputing samples that already have results. Applied after "
            "sharding, so shard membership does not shift as results accumulate."
        ),
    )
    sample_id_column: str = Field(
        "sample_idx",
        description="Column holding the dataset-stable sample id matched against `exclude_sample_ids`.",
    )
    column_mapping: Mapping[str, str] | None = Field(
        None, description="Mapping of the column names in the dataset to the expected column names in the sample class"
    )
    column_transforms: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = Field(
        None,
        description=(
            "Transformations to apply to the columns in the dataset. The key in here are post column mapping names therefore, the expected column names are used."
            "The function should take a row as input and return a row with the transformed column."
            "The function signature should be `def transform(row: dict[str, Any]) -> dict[str, Any]` where the key is the post column mapping name and the value is the transformed column."
        ),
    )

    def load(self) -> HfDataset:
        """Load dataset from config.

        Auto-detects if dataset_id is a local directory path or a HuggingFace dataset ID.
        """
        # Check if dataset_id is a local path
        dataset_path = Path(self.dataset_id)
        if dataset_path.exists() and dataset_path.is_dir():
            return self._load_local()
        else:
            return self._load_huggingface()

    def _load_local(self) -> HfDataset:
        """Load dataset from local directory.

        Supports:
        * HuggingFace ``save_to_disk`` datasets (``dataset_info.json`` /
          ``state.json``, or a DatasetDict with split subdirs), and
        * OpenBench's audio/reference/splits layout via ``local_dataset_loader``.
        """
        dataset_path = Path(self.dataset_id)
        # HF datasets.save_to_disk layout (Dataset or DatasetDict).
        if (dataset_path / "dataset_info.json").exists() or (dataset_path / "state.json").exists() or any(
            (dataset_path / split_name / "dataset_info.json").exists() for split_name in ("train", "test", "validation")
        ):
            from datasets import DatasetDict, load_from_disk

            loaded = load_from_disk(str(dataset_path))
            if isinstance(loaded, DatasetDict):
                split = self.split or next(iter(loaded.keys()))
                ds = loaded[split]
            else:
                ds = loaded
        else:
            from .local_dataset_loader import load_local_dataset

            split = self.split or "test"  # Default split
            ds = load_local_dataset(dataset_dir=dataset_path, split=split)

        return self._postprocess(ds)

    def _load_huggingface(self) -> HfDataset:
        """Load dataset from HuggingFace Hub."""
        # TODO: Add support for streaming datasets
        ds = load_dataset(self.dataset_id, self.subset, split=self.split)
        return self._postprocess(ds)

    def _postprocess(self, ds: HfDataset) -> HfDataset:
        """Apply row selection (sampling, sharding, exclusions) then column fixups."""
        if self.num_samples is not None:
            ds = ds.take(self.num_samples)

        if self.num_shards is not None and self.num_shards > 1:
            if not 0 <= self.shard_index < self.num_shards:
                raise ValueError(f"shard_index must be in [0, {self.num_shards}), got {self.shard_index}")
            total = len(ds)
            # contiguous=False keeps the interleaved ds[shard_index::num_shards]
            # assignment, which balances cost across shards.
            ds = ds.shard(num_shards=self.num_shards, index=self.shard_index, contiguous=False)
            logger.info(f"Shard {self.shard_index}/{self.num_shards}: kept {len(ds)} of {total} rows (interleaved)")

        if self.exclude_sample_ids:
            if self.sample_id_column not in ds.column_names:
                raise ValueError(
                    f"exclude_sample_ids needs column {self.sample_id_column!r}, "
                    f"but the dataset only has {ds.column_names}"
                )
            excluded = self.exclude_sample_ids
            before = len(ds)
            # input_columns keeps the filter from decoding the audio columns.
            ds = ds.filter(lambda sample_id: str(sample_id) not in excluded, input_columns=self.sample_id_column)
            logger.info(f"Excluded {before - len(ds)} already-completed rows, {len(ds)} left to evaluate")

        if self.column_mapping is not None:
            ds = ds.rename_columns(self.column_mapping)

        if self.column_transforms is not None:
            for col, transform in self.column_transforms.items():
                ds = ds.map(transform)

        return ds


class BaseSample(BaseModel, Generic[ReferenceType, ExtraInfoType]):
    """Base class for all sample types with common audio-related functionality."""

    audio_name: str = Field(..., description="The name of the audio file")
    waveform: np.ndarray = Field(..., description="The audio waveform as a numpy array with shape (n_samples,)")
    sample_rate: int = Field(..., description="The sample rate of the audio waveform")
    reference: ReferenceType = Field(..., description="The ground truth object conforming to PredictionProtocol")
    extra_info: ExtraInfoType = Field(default_factory=dict, description="Additional dataset-specific information")

    def get_audio_duration(self) -> float:
        """Calculate audio duration in seconds."""
        return len(self.waveform) / self.sample_rate

    def save_audio(self, output_dir: str | Path) -> Path:
        """Save audio waveform to file."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.audio_name}.flac"
        logger.info(f"Saving audio to {output_path}")
        sf.write(output_path, self.waveform, self.sample_rate)
        return output_path

    class Config:
        arbitrary_types_allowed = True


# TODO: Add support for datasets from local files
class BaseDataset(ABC, Generic[SampleType]):
    """Base class for all dataset types with common functionality."""

    # These MUST be defined in each subclass
    _expected_columns: list[str] = None
    _sample_class: type[SampleType] = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure subclasses define required class attributes."""
        super().__init_subclass__(**kwargs)
        if cls._expected_columns is None:
            raise ValueError(f"Dataset {cls.__name__} must define _expected_columns class attribute")
        if cls._sample_class is None:
            raise ValueError(f"Dataset {cls.__name__} must define _sample_class class attribute")

    def __init__(self, ds: HfDataset) -> None:
        if self._expected_columns is None:
            raise ValueError(f"Dataset {self.__class__.__name__} must define _expected_columns class attribute")
        if self._sample_class is None:
            raise ValueError(f"Dataset {self.__class__.__name__} must define _sample_class class attribute")
        validate_hf_dataset_schema(ds, self._expected_columns)
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(dataset_id={self.organization}/{self.dataset_name}, subset={self.subset}, split={self.split}, num_samples={len(self)})"

    def __getitem__(self, idx: int) -> SampleType:
        """Get a sample by index - concrete implementation using prepare_sample."""
        row = self.ds[idx]
        row["idx"] = idx

        audio_name, waveform, sample_rate = self._extract_audio_info(row)
        reference, extra_info = self.prepare_sample(row)

        return self._create_sample(
            audio_name=audio_name,
            waveform=waveform,
            sample_rate=sample_rate,
            reference=reference,
            extra_info=extra_info,
        )

    @abstractmethod
    def prepare_sample(self, row: dict) -> tuple[ReferenceType, ExtraInfoType]:
        """Prepare the reference and extra_info from dataset row.

        Returns:
            tuple: (reference, extra_info) where reference conforms to PredictionProtocol
                   and extra_info contains dataset-specific metadata
        """
        pass

    def _create_sample(
        self,
        audio_name: str,
        waveform: np.ndarray,
        sample_rate: int,
        reference: ReferenceType,
        extra_info: ExtraInfoType,
    ) -> SampleType:
        """Create the specific sample type using the class-defined sample class."""
        return self._sample_class(
            audio_name=audio_name,
            waveform=waveform,
            sample_rate=sample_rate,
            reference=reference,
            extra_info=extra_info,
        )

    def _extract_audio_info(self, row: dict) -> tuple[str, np.ndarray, int]:
        """Extract common audio information from dataset row."""
        audio = row["audio"]
        audio_name = f"sample_{row['idx']}"
        if "path" in audio and audio["path"] is not None:
            audio_name = Path(audio["path"]).stem
        return audio_name, audio["array"], audio["sampling_rate"]

    # Shared properties
    @property
    def dataset_name(self) -> str:
        return self.ds.info.dataset_name

    @property
    def subset(self) -> str:
        return self.ds.config_name

    @property
    def split(self) -> str:
        return self.ds.split

    @property
    def organization(self) -> str:
        """Get dataset organization. For local datasets, returns 'local'."""
        if not self.ds.info.download_checksums:
            return "local"

        download_url = list(self.ds.info.download_checksums.keys())[0]
        parsed_url = download_url.split("hf://datasets/")[-1]
        return parsed_url.split("/")[0]

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "BaseDataset":
        """Create dataset from configuration."""
        ds = config.load()
        return cls(ds)
