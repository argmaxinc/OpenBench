# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Any

from datasets import Audio as HfAudio
from typing_extensions import NotRequired, TypedDict

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


DEFAULT_SAMPLE_RATE = 16000


class StreamingTranscriptionRow(TypedDict):
    """Expected row structure for streaming transcription datasets."""

    audio: HfAudio  # HF Audio object
    text: str
    word_detail: NotRequired[list[dict[str, float]]]  # List of dicts with "start" and "stop" keys


# Although the prediction of streaming transcription pipelines is a StreamingTranscript object
# we use the Transcript object as the reference since most of attributes from StreamingTranscript
# are produced only at inference without any reference.
class StreamingSample(BaseSample[Transcript, dict[str, Any]]):
    """Streaming transcription sample for real-time transcription tasks."""

    pass


class StreamingDataset(BaseDataset[StreamingSample]):
    """Dataset for streaming transcription pipelines."""

    _expected_columns = ["audio", "text"]
    _sample_class = StreamingSample

    def prepare_sample(self, row: StreamingTranscriptionRow) -> tuple[Transcript, dict[str, Any]]:
        """Prepare streaming transcript and extra info from dataset row."""
        transcript_text = row["text"].split()
        word_timestamps = row.get("word_detail")
        extra_info: dict[str, Any] = {}

        if word_timestamps is None:
            reference = Transcript.from_words_info(words=transcript_text, start=None, end=None, speaker=None)
            return reference, extra_info

        # TODO: compatible datasets should have word timestamps in seconds
        word_timestamps_start = [w["start"] / DEFAULT_SAMPLE_RATE for w in word_timestamps]
        word_timestamps_end = [w["stop"] / DEFAULT_SAMPLE_RATE for w in word_timestamps]

        reference = Transcript.from_words_info(
            words=transcript_text, start=word_timestamps_start, end=word_timestamps_end, speaker=None
        )
        return reference, extra_info
