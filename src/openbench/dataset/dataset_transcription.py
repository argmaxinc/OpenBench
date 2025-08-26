# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing_extensions import TypedDict

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class TranscriptionExtraInfo(TypedDict, total=False):
    """Extra info for transcription samples."""

    language: str


class TranscriptionSample(BaseSample[Transcript, TranscriptionExtraInfo]):
    """Transcription sample for pure transcription tasks."""

    @property
    def language(self) -> str | None:
        """Convenience property to access language from extra_info."""
        return self.extra_info.get("language")


class TranscriptionDataset(BaseDataset[TranscriptionSample]):
    """Dataset for transcription pipelines."""

    _expected_columns = ["audio", "transcript"]
    _sample_class = TranscriptionSample

    def prepare_sample(self, row: dict) -> tuple[Transcript, TranscriptionExtraInfo]:
        """Prepare transcript and extra info from dataset row."""
        transcript_words = row["transcript"]
        reference = Transcript.from_words_info(
            words=transcript_words,
            start=row.get("word_timestamps_start"),
            end=row.get("word_timestamps_end"),
            speaker=None,  # No speakers for pure transcription
        )
        extra_info: TranscriptionExtraInfo = {}
        if "language" in row:
            extra_info["language"] = row["language"]
        return reference, extra_info
