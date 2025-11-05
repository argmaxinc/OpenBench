import os
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from openai.types.audio import TranscriptionDiarized, TranscriptionVerbose


class OpenAIApi:
    def __init__(self, model: Literal["whisper-1", "gpt-4o-transcribe-diarize"] = "whisper-1"):
        self.model = model

        # Check that the API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("`OPENAI_API_KEY` is not set")

        self.client = OpenAI()

    @property
    def is_diarized(self) -> bool:
        return self.model == "gpt-4o-transcribe-diarize"

    def get_transcription_kwargs(self) -> dict[str, Any]:
        # When using the diarized version we need to set response_format to diarized_json and chunking_strategy to auto or vad
        # see https://platform.openai.com/docs/guides/speech-to-text#speaker-diarization
        if self.model == "gpt-4o-transcribe-diarize":
            return {
                "model": self.model,
                "response_format": "diarized_json",
                "chunking_strategy": "auto",
            }

        return {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }

    def transcribe(
        self, audio_path: Path | str, prompt: str | None = None
    ) -> TranscriptionVerbose | TranscriptionDiarized:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        kwargs = self.get_transcription_kwargs()
        with audio_path.open("rb") as audio_file:
            # Use the exact API call from your instructions
            kwargs["file"] = audio_file

            if prompt is not None:
                kwargs["prompt"] = prompt

            response = self.client.audio.transcriptions.create(**kwargs)

        return response
