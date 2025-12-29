from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from deepgram import PrerecordedOptions
from pydantic import Field

from ...dataset import DiarizationSample
from ...engine import DeepgramApi, DeepgramApiResponse
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import OrchestrationConfig, OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class DeepgramOrchestrationPipelineConfig(OrchestrationConfig):
    model_version: str = Field(
        default="nova-3",
        description="The version of the Deepgram model to use",
    )


@register_pipeline
class DeepgramOrchestrationPipeline(Pipeline):
    _config_class = DeepgramOrchestrationPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[Path], DeepgramApiResponse]:
        # Create base API with auto language detection
        base_api = DeepgramApi(
            options=PrerecordedOptions(
                model=self.config.model_version, smart_format=True, diarize=True, detect_language=True
            )
        )

        def transcribe(audio_path: Path) -> DeepgramApiResponse:
            # Use language-specific API if language is set, otherwise use base API
            if self.current_language:
                api = DeepgramApi(
                    options=PrerecordedOptions(
                        model=self.config.model_version,
                        smart_format=True,
                        diarize=True,
                        detect_language=False,
                        language=self.current_language,
                    )
                )
            else:
                api = base_api

            response = api.transcribe(audio_path)
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: DiarizationSample) -> Path:
        # Extract language if force_language is enabled
        self.current_language = None
        if self.config.force_language:
            self.current_language = input_sample.extra_info.get("language", None)

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: DeepgramApiResponse) -> OrchestrationOutput:
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
