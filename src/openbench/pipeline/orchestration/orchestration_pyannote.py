# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""PyannoteAI orchestration pipeline (diarization + transcription with speaker attribution)."""

from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import OrchestrationSample
from ...engine import PyannoteAIApi, PyannoteApiOrchestrationOutput
from ...pipeline_prediction import Transcript
from ..base import Pipeline, PipelineType, register_pipeline
from .common import OrchestrationConfig, OrchestrationOutput


__all__ = ["PyannoteOrchestrationPipeline", "PyannoteOrchestrationPipelineConfig"]

logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class PyannoteOrchestrationPipelineConfig(OrchestrationConfig):
    """Configuration for PyannoteAI orchestration pipeline."""

    timeout: int = Field(
        default=1800,
        description="Timeout for the orchestration job in seconds",
    )
    request_buffer: int = Field(
        default=30,
        description="Buffer for the request rate limit",
    )


@register_pipeline
class PyannoteOrchestrationPipeline(Pipeline):
    """
    PyannoteAI orchestration pipeline.

    Uses the PyannoteAI API with transcription enabled to get both diarization
    and speaker-attributed transcription results.
    """

    _config_class = PyannoteOrchestrationPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(
        self,
    ) -> Callable[[Path], PyannoteApiOrchestrationOutput]:
        api = PyannoteAIApi(
            timeout=self.config.timeout,
            request_buffer=self.config.request_buffer,
            transcription=True,
        )

        def orchestrate(audio_path: Path) -> PyannoteApiOrchestrationOutput:
            response = api(audio_path=str(audio_path))
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return orchestrate

    def parse_input(self, input_sample: OrchestrationSample) -> Path:
        """Save audio to temporary directory for processing."""
        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: PyannoteApiOrchestrationOutput) -> OrchestrationOutput:
        """
        Parse the PyannoteAI response into a Transcript with speaker attribution.

        Uses word-level transcription to get precise word timings with speaker labels.
        """
        # Extract words from word-level transcription with speaker attribution
        transcript = Transcript.from_words_info(
            words=[word.text for word in output.output.word_level_transcription],
            start=[word.start for word in output.output.word_level_transcription],
            end=[word.end for word in output.output.word_level_transcription],
            speaker=[word.speaker for word in output.output.word_level_transcription],
        )

        return OrchestrationOutput(
            prediction=transcript,
            diarization_output=None,
            transcription_output=None,
        )
