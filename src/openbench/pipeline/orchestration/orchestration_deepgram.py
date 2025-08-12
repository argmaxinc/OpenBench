from pathlib import Path
from typing import Callable

from deepgram import PrerecordedOptions
from pydantic import Field

from ...dataset import DiarizationSample
from ...engine import DeepgramApi, DeepgramApiResponse
from ...pipeline import Pipeline, PipelineConfig, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import OrchestrationOutput


TEMP_AUDIO_DIR = Path("temp_audio_dir")


class DeepgramOrchestrationPipelineConfig(PipelineConfig):
    model_version: str = Field(
        default="nova-3",
        description="The version of the Deepgram model to use",
    )


@register_pipeline
class DeepgramOrchestrationPipeline(Pipeline):
    _config_class = DeepgramOrchestrationPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[Path], DeepgramApiResponse]:
        deepgram_api = DeepgramApi(
            options=PrerecordedOptions(
                model=self.config.model_version, smart_format=True, diarize=True, detect_language=True
            )
        )

        def transcribe(audio_path: Path) -> DeepgramApiResponse:
            response = deepgram_api.transcribe(audio_path)
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: DiarizationSample) -> Path:
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
