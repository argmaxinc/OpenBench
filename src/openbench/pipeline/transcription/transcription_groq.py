import os
from pathlib import Path

import groq
from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import TranscriptionSample
from ...pipeline_prediction import Transcript
from ..base import Pipeline, PipelineType, register_pipeline
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("./temp_audio")


class GroqEngine:
    def __init__(self, model_id: str, temperature: float = 0.0, hint_language: bool = False) -> None:
        if "GROQ_API_KEY" not in os.environ:
            raise ValueError("`GROQ_API_KEY` is not set. Please set it in the environment variables.")

        self.client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model_id = model_id
        self.temperature = temperature
        self.hint_language = hint_language

    def __call__(self, inputs: TranscriptionSample) -> str:
        audio_file = inputs.save_audio(TEMP_AUDIO_DIR)
        language = None
        if self.hint_language and inputs.language is not None:
            language = inputs.language

        with open(audio_file, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                file=f,
                model=self.model_id,
                response_format="text",
                language=language,
                temperature=self.temperature,
            )

        return transcription


class GroqTranscriptionConfig(TranscriptionConfig):
    model_id: str = Field(
        ...,
        description="The ID of the Groq model to use",
    )
    temperature: float = Field(
        default=0.0,
        description="The temperature to use for the transcription",
    )


@register_pipeline
class GroqTranscriptionPipeline(Pipeline):
    _config_class = GroqTranscriptionConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> GroqEngine:
        return GroqEngine(
            model_id=self.config.model_id,
            temperature=self.config.temperature,
            hint_language=self.config.force_language,
        )

    def parse_input(self, input_sample: TranscriptionSample) -> TranscriptionSample:
        return input_sample

    def parse_output(self, output: str) -> TranscriptionOutput:
        return TranscriptionOutput(prediction=Transcript.from_words_info(words=output.split()))
