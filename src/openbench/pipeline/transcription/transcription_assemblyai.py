import os
import time
from pathlib import Path
from typing import Callable

import requests
from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import TranscriptionSample
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class AssemblyAIApi:
    def __init__(self, cfg) -> None:
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        assert self.api_key is not None, "Please set ASSEMBLYAI_API_KEY in environment"

        self.base_url = "https://api.assemblyai.com"
        self.headers = {"authorization": self.api_key}
        self.model_version = cfg.model_version

    def transcribe(self, audio_path: Path, keywords=None):
        """Transcribe audio file using AssemblyAI REST API."""

        with open(audio_path, "rb") as f:
            upload_response = requests.post(f"{self.base_url}/v2/upload", headers=self.headers, data=f)

        if upload_response.status_code != 200:
            raise RuntimeError(f"Upload failed: {upload_response.status_code}, {upload_response.text}")

        upload_url = upload_response.json()["upload_url"]

        data = {"audio_url": upload_url, "speech_model": self.model_version}

        # Add keywords if provided
        if keywords:
            data["keyterms_prompt"] = keywords
            logger.debug(f"Using keywords: {keywords}")

        # Submit transcription request
        url = f"{self.base_url}/v2/transcript"
        response = requests.post(url, json=data, headers=self.headers)

        if response.status_code != 200:
            raise RuntimeError(f"Transcription request failed: {response.status_code}, {response.text}")

        transcript_id = response.json()["id"]
        polling_endpoint = f"{self.base_url}/v2/transcript/{transcript_id}"

        while True:
            transcription_result = requests.get(polling_endpoint, headers=self.headers).json()

            if transcription_result["status"] == "completed":
                return transcription_result["text"]
            elif transcription_result["status"] == "error":
                raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
            else:
                logger.debug(f"Transcription status: {transcription_result['status']}, waiting...")
                time.sleep(3)


class AssemblyAITranscriptionPipelineConfig(TranscriptionConfig):
    model_version: str = Field(
        default="universal",
        description="The version of the AssemblyAI speech model to use",
    )


@register_pipeline
class AssemblyAITranscriptionPipeline(Pipeline):
    _config_class = AssemblyAITranscriptionPipelineConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], str]:
        assemblyai_api = AssemblyAIApi(self.config)

        def transcribe(audio_path: Path) -> str:
            response = assemblyai_api.transcribe(audio_path, keywords=self.current_keywords)
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
