import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, model_validator


class OpenAIApiResponse(BaseModel):
    words: list[str]
    start: list[float]
    end: list[float]

    @property
    def transcript(self) -> str:
        return " ".join(self.words)

    @model_validator(mode="after")
    def validate_lengths(self) -> "OpenAIApiResponse":
        if len(self.words) != len(self.start) or len(self.words) != len(self.end):
            raise ValueError("All lists must be of the same length")
        return self


class OpenAIApi:
    def __init__(self, model: str = "whisper-1"):
        self.model = model

        # Check that the API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("`OPENAI_API_KEY` is not set")

        self.client = OpenAI()

    def transcribe(self, audio_path: Path | str, prompt: str | None = None) -> OpenAIApiResponse:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        with audio_path.open("rb") as audio_file:
            # Use the exact API call from your instructions
            kwargs = {
                "model": self.model,
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"],
            }

            if prompt is not None:
                kwargs["prompt"] = prompt

            response = self.client.audio.transcriptions.create(**kwargs)

        # Extract words and timestamps from verbose_json response
        words = []
        start_times = []
        end_times = []
        speakers = []
        if hasattr(response, "words") and response.words:
            # Use word-level timestamps from OpenAI
            for word_info in response.words:
                # Handle both dict and object formats
                if isinstance(word_info, dict):
                    words.append(word_info["word"].strip())
                    start_times.append(float(word_info["start"]))
                    end_times.append(float(word_info["end"]))
                else:
                    words.append(word_info.word.strip())
                    start_times.append(float(word_info.start))
                    end_times.append(float(word_info.end))
                speakers.append("0")  # OpenAI doesn't provide speaker info
        else:
            # Fallback: split transcript into words without timestamps
            transcript_words = response.text.strip().split()
            for word in transcript_words:
                words.append(word)
                start_times.append(0.0)
                end_times.append(0.0)
                speakers.append("0")

        return OpenAIApiResponse(
            words=words,
            start=start_times,
            end=end_times,
        )
