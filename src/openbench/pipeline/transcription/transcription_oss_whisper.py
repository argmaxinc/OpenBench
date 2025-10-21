# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import TranscriptionSample
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput
import whisper


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class WhisperOSSApi:
    def __init__(
        self, model_version: str = "base", device: str | None = None
    ):
        """
        Initialize OpenAI Whisper (open-source) engine.

        Args:
            model_version: Whisper model size
                (tiny, base, small, medium, large, turbo)
            device: Device to use for inference (cuda, cpu, mps).
                If None, auto-detect.
        """

        self.model_version = model_version

        # Auto-detect device if not specified
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        logger.info(
            f"Loading Whisper model '{model_version}' "
            f"on device '{self.device}'"
        )

        # Load the model with download_root to suppress progress bars
        self.model = whisper.load_model(
            model_version, device=self.device, download_root=None
        )

    def transcribe(
        self,
        audio_path: Path,
        prompt: str | None = None,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe an audio file using OpenAI Whisper (open-source).

        Args:
            audio_path: Path to the audio file
            prompt: Optional initial prompt for the model
            language: Optional language code (e.g., 'en', 'es')

        Returns:
            Dictionary with transcription results including words and
            timestamps
        """
        # Prepare transcription options
        transcribe_options = {
            "word_timestamps": True,
            "verbose": None,  # Suppress all output including progress bars
            "fp16": False,  # Avoid fp16 warnings and issues
        }

        if prompt is not None:
            transcribe_options["initial_prompt"] = prompt
            logger.debug(f"Using prompt: {prompt}")

        if language is not None:
            transcribe_options["language"] = language
            logger.debug(f"Using language: {language}")

        # Transcribe
        result = self.model.transcribe(
            str(audio_path), **transcribe_options
        )

        # Extract words and timestamps
        words = []
        start_times = []
        end_times = []

        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        words.append(word_info["word"].strip())
                        start_times.append(float(word_info["start"]))
                        end_times.append(float(word_info["end"]))

        # Fallback: if no word-level timestamps, split transcript
        if not words and "text" in result:
            transcript_words = result["text"].strip().split()
            for word in transcript_words:
                words.append(word)
                start_times.append(0.0)
                end_times.append(0.0)

        return {
            "words": words,
            "start": start_times,
            "end": end_times,
        }


class WhisperOSSTranscriptionPipelineConfig(TranscriptionConfig):
    model_version: str = Field(
        default="base",
        description=(
            "The version of the Whisper model to use "
            "(tiny, base, small, medium, large, turbo)"
        ),
    )
    device: str | None = Field(
        default=None,
        description=(
            "Device to use for inference (cuda, cpu, mps). "
        ),
    )


@register_pipeline
class WhisperOSSTranscriptionPipeline(Pipeline):
    _config_class = WhisperOSSTranscriptionPipelineConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], dict]:
        whisper_api = WhisperOSSApi(
            model_version=self.config.model_version, device=self.config.device
        )

        def transcribe(audio_path: Path) -> dict:
            language = None
            if self.config.force_language:
                language = self.current_language

            response = whisper_api.transcribe(
                audio_path,
                prompt=self.current_keywords_prompt,
                language=language,
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> Path:
        """Override to extract keywords from sample before processing."""
        # Extract keywords from sample's extra_info if flag is enabled
        self.current_keywords_prompt = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                # Format keywords as comma-separated prompt for Whisper
                self.current_keywords_prompt = ", ".join(keywords)

        # Extract language if force_language is enabled
        self.current_language = None
        if self.config.force_language:
            self.current_language = input_sample.extra_info.get(
                "language", None
            )

        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: dict) -> TranscriptionOutput:
        return TranscriptionOutput(
            prediction=Transcript.from_words_info(
                words=output["words"],
                start=output["start"],
                end=output["end"],
            )
        )
