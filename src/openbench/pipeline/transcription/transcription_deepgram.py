from pathlib import Path
from typing import Callable

from deepgram import PrerecordedOptions
from pydantic import Field

from ...engine import DeepgramApi, DeepgramApiResponse
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
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], DeepgramApiResponse]:
        # Create base API without language detection
        base_api = DeepgramApi(
            options=PrerecordedOptions(
                model=self.config.model_version, smart_format=True, detect_language=not self.config.force_language
            )
        )

        def transcribe(audio_path: Path) -> DeepgramApiResponse:
            # Use language-specific API if language is set, otherwise use base API
            if self.current_language:
                base_api.set_language(self.current_language)

            response = base_api.transcribe(audio_path, keyterm=self.current_keywords)
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample) -> Path:
        """Override to extract keywords and language from sample before processing."""
        self.current_keywords = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                # Add + between keywords for Deepgram URL
                self.current_keywords = "+".join(keywords)

        # Extract language if force_language is enabled
        self.current_language = None
        if self.config.force_language:
            self.current_language = input_sample.language

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: DeepgramApiResponse) -> TranscriptionOutput:
        return TranscriptionOutput(
            prediction=Transcript.from_words_info(
                words=output.words,
                speaker=output.speakers,
                start=output.start,
                end=output.end,
            )
        )
