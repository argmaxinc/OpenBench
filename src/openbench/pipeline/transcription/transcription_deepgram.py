from pathlib import Path
from typing import Callable

from deepgram import PrerecordedOptions
from pydantic import Field

from ...dataset import DiarizationSample
from ...engine import DeepgramApi, DeepgramApiInput, DeepgramApiResponse
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput


TEMP_AUDIO_DIR = Path("temp_audio_dir")


class DeepgramTranscriptionPipelineConfig(TranscriptionConfig):
    model_version: str = Field(
        default="nova-3",
        description="The version of the Deepgram model to use",
    )


@register_pipeline
class DeepgramTranscriptionPipeline(Pipeline):
    _config_class = DeepgramTranscriptionPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[DeepgramApiInput], DeepgramApiResponse]:
        deepgram_api = DeepgramApi(
            options=PrerecordedOptions(model=self.config.model_version, smart_format=True, detect_language=True)
        )

        def transcribe(inputs: DeepgramApiInput) -> DeepgramApiResponse:
            response = deepgram_api.transcribe(inputs)
            # Remove temporary audio path
            inputs.audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: DiarizationSample) -> DeepgramApiInput:
        audio_path = input_sample.save_audio(TEMP_AUDIO_DIR)
        keywords = input_sample.keywords if self.config.use_keywords else None

        return DeepgramApiInput(
            audio_path=audio_path,
            keywords=keywords,
        )

    def parse_output(self, output: DeepgramApiResponse) -> TranscriptionOutput:
        return TranscriptionOutput(
            prediction=Transcript.from_words_info(
                words=output.words,
                speaker=output.speakers,
                start=output.start,
                end=output.end,
            )
        )
