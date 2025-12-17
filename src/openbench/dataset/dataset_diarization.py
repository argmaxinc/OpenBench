# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

# Importing from typing_extensions is required for Python <3.12 to use TypedDict with pydantic
from datasets import Audio as HfAudio
from pyannote.core import Segment, Timeline
from typing_extensions import NotRequired, TypedDict

from ..pipeline_prediction import DiarizationAnnotation
from .dataset_base import BaseDataset, BaseSample


class DiarizationExtraInfo(TypedDict, total=False):
    """Extra info for diarization samples."""

    uem: Timeline


class DiarizationRow(TypedDict):
    """Expected row structure for diarization datasets."""

    audio: HfAudio  # HF Audio object
    timestamps_start: list[float]
    timestamps_end: list[float]
    speakers: list[str]
    uem_timestamps: NotRequired[list[tuple[float, float]]]


class DiarizationSample(BaseSample[DiarizationAnnotation, DiarizationExtraInfo]):
    """Diarization sample with UEM convenience property."""

    @property
    def uem(self) -> Timeline | None:
        """Convenience property to access UEM from extra_info."""
        return self.extra_info.get("uem")


class DiarizationDataset(BaseDataset[DiarizationSample]):
    """Dataset for diarization pipelines."""

    _expected_columns = ["audio", "timestamps_start", "timestamps_end", "speakers"]
    _sample_class = DiarizationSample

    def prepare_sample(self, row: DiarizationRow) -> tuple[DiarizationAnnotation, DiarizationExtraInfo]:
        """Prepare diarization annotation and extra info from dataset row."""
        # Prepare reference
        timestamps_start = row["timestamps_start"]
        timestamps_end = row["timestamps_end"]
        speakers = row["speakers"]

        annotation = DiarizationAnnotation()
        for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
            segment = Segment(start, end)
            annotation[segment] = speaker

        # Prepare extra info
        extra_info: DiarizationExtraInfo = {}
        if "uem_timestamps" in row:
            uem_timestamps = row["uem_timestamps"]
            uem = Timeline()
            for start, end in uem_timestamps:
                uem.add(Segment(start, end))
            extra_info["uem"] = uem

        return annotation, extra_info
