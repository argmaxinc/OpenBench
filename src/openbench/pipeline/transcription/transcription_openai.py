from pathlib import Path
from typing import Callable

from pydantic import Field

from ...engine import OpenAIApi, OpenAIApiResponse
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput


TEMP_AUDIO_DIR = Path("temp_audio_dir")


class OpenAITranscriptionPipelineConfig(TranscriptionConfig):
    model_version: str = Field(
        default="whisper-1",
        description="The version of the OpenAI model to use",
    )


@register_pipeline
class OpenAITranscriptionPipeline(Pipeline):
    _config_class = OpenAITranscriptionPipelineConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], OpenAIApiResponse]:
        openai_api = OpenAIApi(model=self.config.model_version)

        def transcribe(audio_path: Path) -> OpenAIApiResponse:
            response = openai_api.transcribe(
                audio_path,
                prompt=self.current_keywords_prompt,
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample) -> Path:
        """Override to extract keywords from sample before processing."""
        # Extract keywords from sample's extra_info if flag is enabled
        self.current_keywords_prompt = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get('dictionary', [])
            if keywords:
                # Format keywords as comma-separated prompt for OpenAI
                self.current_keywords_prompt = ", ".join(keywords)

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: OpenAIApiResponse) -> TranscriptionOutput:
        return TranscriptionOutput(
            prediction=Transcript.from_words_info(
                words=output.words,
                start=output.start,
                end=output.end,
            )
        )
