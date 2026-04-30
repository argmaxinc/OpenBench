# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import numpy as np
from typing_extensions import TypedDict

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class SpeechGenerationExtraInfo(TypedDict, total=False):
    """Extra info for speech generation samples."""

    language: str


class SpeechGenerationRow(TypedDict):
    """Expected row structure for speech generation.

    Requires 'text' (the prompt string). No audio needed.
    """

    text: str


class SpeechGenerationSample(BaseSample[Transcript, SpeechGenerationExtraInfo]):
    """Sample for speech-generation tasks.

    The reference `Transcript` is constructed from the prompt text. The
    pipeline synthesizes audio from this prompt and returns a
    `GeneratedAudio` prediction; the WER metric transcribes that audio
    and compares against this reference.
    """

    @property
    def text(self) -> str:
        """The original text prompt."""
        return self.reference.get_transcript_string()


class SpeechGenerationDataset(BaseDataset[SpeechGenerationSample]):
    """Dataset for speech-generation pipelines.

    Expects column: 'text' (the prompt string). No audio column is
    required — audio is produced by the pipeline. A dummy waveform is
    supplied to satisfy the base sample structure but is ignored
    everywhere downstream; the runner reads the real generated-audio
    duration off the pipeline output.
    """

    _expected_columns = ["text"]
    _sample_class = SpeechGenerationSample

    def _extract_audio_info(self, row: dict) -> tuple[str, np.ndarray, int]:
        """Provide a placeholder waveform; speech-generation has no input audio."""
        audio_name = f"sample_{row['idx']}"
        # Use audio_name from the row if available
        if "audio_name" in row and row["audio_name"]:
            audio_name = str(row["audio_name"])
        dummy_waveform = np.zeros(1, dtype=np.float32)
        dummy_sample_rate = 16000
        return audio_name, dummy_waveform, dummy_sample_rate

    def prepare_sample(self, row: SpeechGenerationRow) -> tuple[Transcript, SpeechGenerationExtraInfo]:
        """Build the reference transcript from the prompt text."""
        text = row["text"]
        words = text.split()
        reference = Transcript.from_words_info(
            words=words,
        )

        extra_info: SpeechGenerationExtraInfo = {}
        if "language" in row:
            extra_info["language"] = row["language"]

        return reference, extra_info
