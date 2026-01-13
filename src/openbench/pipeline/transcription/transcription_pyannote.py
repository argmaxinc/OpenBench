# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""PyannoteAI transcription pipeline (ignores speaker attribution)."""

from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import TranscriptionSample
from ...engine import PyannoteAIApi, PyannoteApiOrchestrationOutput
from ...pipeline_prediction import Transcript
from ..base import Pipeline, PipelineType, register_pipeline
from .common import TranscriptionConfig, TranscriptionOutput


__all__ = ["PyannoteTranscriptionPipeline", "PyannoteTranscriptionPipelineConfig"]

logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class PyannoteTranscriptionPipelineConfig(TranscriptionConfig):
    """Configuration for PyannoteAI transcription pipeline."""

    timeout: int = Field(
        default=1800,
        description="Timeout for the transcription job in seconds",
    )
    request_buffer: int = Field(
        default=30,
        description="Buffer for the request rate limit",
    )


@register_pipeline
class PyannoteTranscriptionPipeline(Pipeline):
    """
    PyannoteAI transcription pipeline.

    Uses the PyannoteAI API with transcription enabled, but ignores speaker
    attribution in the output. This is useful for datasets that don't have
    speaker labels.
    """

    _config_class = PyannoteTranscriptionPipelineConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(
        self,
    ) -> Callable[[Path], PyannoteApiOrchestrationOutput]:
        api = PyannoteAIApi(
            timeout=self.config.timeout,
            request_buffer=self.config.request_buffer,
            transcription=True,
        )

        def transcribe(audio_path: Path) -> PyannoteApiOrchestrationOutput:
            response = api(audio_path=str(audio_path))
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> Path:
        """Save audio to temporary directory for processing."""
        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: PyannoteApiOrchestrationOutput) -> TranscriptionOutput:
        """
        Parse the PyannoteAI response into a Transcript without speaker attribution.

        The word-level transcription contains speaker information, but we ignore it
        here since this is a transcription-only pipeline.
        """
        # Extract words from word-level transcription, ignoring speaker info
        transcript = Transcript.from_words_info(
            words=[word.text for word in output.output.word_level_transcription],
            start=[word.start for word in output.output.word_level_transcription],
            end=[word.end for word in output.output.word_level_transcription],
            speaker=None,  # Ignore speaker attribution for transcription pipeline
        )

        return TranscriptionOutput(prediction=transcript)
