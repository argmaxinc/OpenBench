import os
from pathlib import Path

from argmaxtools.utils import get_logger
from elevenlabs.client import ElevenLabs
from pydantic import BaseModel, model_validator


logger = get_logger(__name__)


class ElevenLabsApiResponse(BaseModel):
    """Response from ElevenLabs speech-to-text API."""

    words: list[str]
    speakers: list[str]
    start: list[float]
    end: list[float]

    @property
    def transcript(self) -> str:
        return " ".join(self.words)

    @model_validator(mode="after")
    def validate_lengths(self) -> "ElevenLabsApiResponse":
        if (
            len(self.words) != len(self.speakers)
            or len(self.words) != len(self.start)
            or len(self.words) != len(self.end)
        ):
            raise ValueError("All lists must be of the same length")
        return self


class ElevenLabsApi:
    """ElevenLabs Speech-to-Text API wrapper."""

    def __init__(
        self,
        model_id: str = "scribe_v2",
        timeout: float = 300,
    ):
        self.model_id = model_id
        self.timeout = timeout

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("`ELEVENLABS_API_KEY` is not set")

        self.client = ElevenLabs(api_key=api_key, timeout=timeout)

    def transcribe(
        self,
        audio_path: Path | str,
        keyterms: list[str] | None = None,
        language_code: str | None = None,
        diarize: bool = False,
        num_speakers: int | None = None,
    ) -> ElevenLabsApiResponse:
        """Transcribe an audio file using ElevenLabs API.

        Args:
            audio_path: Path to the audio file
            keyterms: List of keywords to boost recognition
            language_code: Language code (e.g., 'eng')
            diarize: Whether to enable speaker diarization
            num_speakers: Maximum number of speakers

        Returns:
            ElevenLabsApiResponse with words, speakers, and timestamps
        """
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        with audio_path.open("rb") as f:
            audio_data = f.read()

        kwargs = {
            "model_id": self.model_id,
            "file": audio_data,
        }

        if keyterms:
            kwargs["keyterms"] = keyterms
            logger.debug(f"Using keyterms: {keyterms}")

        if language_code:
            kwargs["language_code"] = language_code
            logger.debug(f"Using language: {language_code}")

        if diarize:
            kwargs["diarize"] = True
            logger.debug("Diarization enabled")

        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
            logger.debug(f"Max speakers: {num_speakers}")

        response = self.client.speech_to_text.convert(**kwargs)

        # ElevenLabs returns whitespace as separate "words" - filter them out
        words = [w for w in response.words if w.text and w.text.strip()]

        return ElevenLabsApiResponse(
            words=[w.text for w in words],
            speakers=[str(w.speaker_id) for w in words],
            start=[float(w.start) for w in words],
            end=[float(w.end) for w in words],
        )
