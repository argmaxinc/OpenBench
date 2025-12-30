# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from datasets import Audio as HfAudio
from pydantic import model_validator
from typing_extensions import NotRequired, TypedDict

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class OrchestrationExtraInfo(TypedDict, total=False):
    """Extra info for orchestration samples."""

    language: str


class OrchestrationRow(TypedDict):
    """Expected row structure for orchestration datasets."""

    audio: HfAudio  # HF Audio object
    transcript: list[str]
    word_speakers: list[str]
    word_timestamps_start: NotRequired[list[float]]
    word_timestamps_end: NotRequired[list[float]]
    language: NotRequired[str]


class OrchestrationSample(BaseSample[Transcript, OrchestrationExtraInfo]):
    """Orchestration sample with speaker validation."""

    @model_validator(mode="after")
    def validate_speaker_labels(self) -> "OrchestrationSample":
        """Ensure transcript has speaker labels."""
        if not self.reference.has_speakers:
            raise ValueError("Orchestration samples require transcript with speaker labels")
        return self

    @property
    def language(self) -> str | None:
        """Convenience property to access language from extra_info."""
        return self.extra_info.get("language")


class OrchestrationDataset(BaseDataset[OrchestrationSample]):
    """Dataset for orchestration pipelines."""

    _expected_columns = ["audio", "transcript", "word_speakers"]
    _sample_class = OrchestrationSample

    def prepare_sample(self, row: OrchestrationRow) -> tuple[Transcript, OrchestrationExtraInfo]:
        """Prepare transcript with speaker labels and extra info from dataset row."""
        transcript_words = row["transcript"]
        word_speakers = row["word_speakers"]

        if len(word_speakers) != len(transcript_words):
            raise ValueError("word_speakers and transcript must have same length")

        reference = Transcript.from_words_info(
            words=transcript_words,
            start=row.get("word_timestamps_start"),
            end=row.get("word_timestamps_end"),
            speaker=word_speakers,
        )
        extra_info: OrchestrationExtraInfo = {}
        if "language" in row:
            extra_info["language"] = row["language"]
        return reference, extra_info
