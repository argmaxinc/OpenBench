# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from datasets import Audio as HfAudio
from typing_extensions import NotRequired, TypedDict

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class TranscriptionExtraInfo(TypedDict, total=False):
    """Extra info for transcription samples."""

    language: str
    dictionary: list[str]


class TranscriptionRow(TypedDict):
    """Expected row structure for transcription datasets."""

    audio: HfAudio  # HF Audio object
    transcript: list[str]
    word_timestamps_start: NotRequired[list[float]]
    word_timestamps_end: NotRequired[list[float]]
    language: NotRequired[str]
    dictionary: NotRequired[list[str]]


class TranscriptionSample(BaseSample[Transcript, TranscriptionExtraInfo]):
    """Transcription sample for transcription tasks with optional keyword support."""

    @property
    def language(self) -> str | None:
        """Convenience property to access language from extra_info."""
        return self.extra_info.get("language")

    @property
    def dictionary(self) -> list[str] | None:
        """Convenience property to access dictionary from extra_info."""
        return self.extra_info.get("dictionary")


class TranscriptionDataset(BaseDataset[TranscriptionSample]):
    """Dataset for transcription pipelines with optional keyword support."""

    _expected_columns = ["audio", "transcript"]
    _sample_class = TranscriptionSample

    def prepare_sample(self, row: TranscriptionRow) -> tuple[Transcript, TranscriptionExtraInfo]:
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
        if "dictionary" in row:
            extra_info["dictionary"] = row["dictionary"]
        return reference, extra_info
