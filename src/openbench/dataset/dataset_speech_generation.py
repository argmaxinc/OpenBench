# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import numpy as np
from pydantic import Field
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


class SpeechGenerationSample(
    BaseSample[Transcript, SpeechGenerationExtraInfo]
):
    """Sample for speech generation tasks.

    The reference Transcript is created from the text
    prompt. The pipeline generates audio from this text
    and transcribes it to compute WER against reference.
    """

    generated_audio_duration: float | None = Field(
        default=None,
        description=(
            "Duration (seconds) of the TTS-generated audio. "
            "Set by the pipeline after generation."
        ),
    )

    def get_audio_duration(self) -> float:
        """Return generated audio duration if available.

        Falls back to the dummy waveform calculation
        when the pipeline hasn't set the real duration yet.
        """
        if self.generated_audio_duration is not None:
            return self.generated_audio_duration
        return super().get_audio_duration()

    @property
    def text(self) -> str:
        """The original text prompt."""
        return self.reference.get_transcript_string()


class SpeechGenerationDataset(
    BaseDataset[SpeechGenerationSample]
):
    """Dataset for speech generation pipelines.

    Expects column: 'text' (the prompt string).
    No audio column is required since audio is generated
    by the pipeline itself.
    """

    _expected_columns = ["text"]
    _sample_class = SpeechGenerationSample

    def _extract_audio_info(
        self, row: dict
    ) -> tuple[str, np.ndarray, int]:
        """Override to provide dummy audio info.

        Speech generation datasets don't have input audio.
        We provide a placeholder waveform so the framework
        sample structure is satisfied. The pipeline ignores
        the waveform entirely.
        """
        audio_name = f"sample_{row['idx']}"
        # Use audio_name from the row if available
        if "audio_name" in row and row["audio_name"]:
            audio_name = str(row["audio_name"])
        dummy_waveform = np.zeros(1, dtype=np.float32)
        dummy_sample_rate = 16000
        return audio_name, dummy_waveform, dummy_sample_rate

    def prepare_sample(
        self, row: SpeechGenerationRow
    ) -> tuple[Transcript, SpeechGenerationExtraInfo]:
        """Prepare reference from dataset row.

        Splits text prompt into words to create the
        reference Transcript.
        """
        text = row["text"]
        words = text.split()
        reference = Transcript.from_words_info(
            words=words,
        )

        extra_info: SpeechGenerationExtraInfo = {}
        if "language" in row:
            extra_info["language"] = row["language"]

        return reference, extra_info
