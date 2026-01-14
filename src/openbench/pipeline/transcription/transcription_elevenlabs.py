from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import TranscriptionSample
from ...engine import ElevenLabsApi, ElevenLabsApiResponse
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class ElevenLabsTranscriptionPipelineConfig(TranscriptionConfig):
    model_id: str = Field(
        default="scribe_v2",
        description="The ElevenLabs speech-to-text model to use",
    )


@register_pipeline
class ElevenLabsTranscriptionPipeline(Pipeline):
    _config_class = ElevenLabsTranscriptionPipelineConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], ElevenLabsApiResponse]:
        api = ElevenLabsApi(model_id=self.config.model_id)

        def transcribe(audio_path: Path) -> ElevenLabsApiResponse:
            response = api.transcribe(
                audio_path=audio_path,
                keyterms=self.current_keywords,
                language_code=self.current_language,
                diarize=False,
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> Path:
        """Override to extract keywords from sample before processing."""
        self.current_keywords = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                self.current_keywords = keywords

        self.current_language = None
        if self.config.force_language:
            self.current_language = input_sample.language

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: ElevenLabsApiResponse) -> TranscriptionOutput:
        return TranscriptionOutput(
            prediction=Transcript.from_words_info(
                words=output.words,
                speaker=None,
                start=output.start,
                end=output.end,
            )
        )
