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
        # TODO: Add support for streaming datasets
        ds = load_dataset(self.dataset_id, self.subset, split=self.split)
        if self.num_samples is not None:
            ds = ds.take(self.num_samples)

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
        download_url = list(self.ds.info.download_checksums.keys())[0]
        parsed_url = download_url.split("hf://datasets/")[-1]
        return parsed_url.split("/")[0]

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "BaseDataset":
        """Create dataset from configuration."""
        ds = config.load()
        return cls(ds)
