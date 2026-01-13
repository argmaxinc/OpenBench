import os
from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from elevenlabs.client import ElevenLabs
from pydantic import Field

from ...dataset import TranscriptionSample
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

    def build_pipeline(self) -> Callable[[Path], str]:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        assert api_key is not None, "Please set ELEVENLABS_API_KEY in environment"

        client = ElevenLabs(api_key=api_key)

        def transcribe(audio_path: Path) -> str:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            kwargs = {
                "file": audio_data,
                "model_id": self.config.model_id,
            }

            # Add keyterms if available (up to 100, max 50 chars each)
            if self.current_keywords:
                # Filter keywords to max 50 chars and limit to 100
                filtered_keywords = [kw[:50] for kw in self.current_keywords[:100]]
                kwargs["keyterms"] = filtered_keywords
                logger.debug(f"Using keyterms: {filtered_keywords}")

            transcription = client.speech_to_text.convert(**kwargs)

            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)

            return transcription.text

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> Path:
        """Override to extract keywords from sample before processing."""
        self.current_keywords = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                self.current_keywords = keywords

        # Warn if force_language is enabled (not currently supported)
        if self.config.force_language:
            logger.warning(
                f"{self.__class__.__name__} does not support language hinting. "
                "The force_language flag will be ignored."
            )

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: str) -> TranscriptionOutput:
        # Split transcript into words
        words = output.split() if output else []
        transcript = Transcript.from_words_info(words=words)
        return TranscriptionOutput(prediction=transcript)

