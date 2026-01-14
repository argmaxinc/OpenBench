from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import OrchestrationSample
from ...engine import ElevenLabsApi, ElevenLabsApiResponse
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import OrchestrationConfig, OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class ElevenLabsOrchestrationPipelineConfig(OrchestrationConfig):
    model_id: str = Field(
        default="scribe_v2",
        description="The ElevenLabs speech-to-text model to use",
    )
    num_speakers: int | None = Field(
        default=None,
        description="Maximum number of speakers (helps with diarization). Max 32.",
    )


@register_pipeline
class ElevenLabsOrchestrationPipeline(Pipeline):
    _config_class = ElevenLabsOrchestrationPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[Path], ElevenLabsApiResponse]:
        api = ElevenLabsApi(model_id=self.config.model_id)

        def orchestrate(audio_path: Path) -> ElevenLabsApiResponse:
            response = api.transcribe(
                audio_path=audio_path,
                language_code=self.current_language,
                diarize=True,
                num_speakers=self.config.num_speakers,
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return orchestrate

    def parse_input(self, input_sample: OrchestrationSample) -> Path:
        """Override to extract language from sample before processing."""
        self.current_language = None
        if self.config.force_language:
            self.current_language = input_sample.language

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: ElevenLabsApiResponse) -> OrchestrationOutput:
        return OrchestrationOutput(
            prediction=Transcript.from_words_info(
                words=output.words,
                speaker=output.speakers,
                start=output.start,
                end=output.end,
            ),
            diarization_output=None,
            transcription_output=None,
        )

