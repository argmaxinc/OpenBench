import os
from pathlib import Path

from deepgram import DeepgramClient, FileSource, PrerecordedOptions, PrerecordedResponse
from pydantic import BaseModel, model_validator


class DeepgramApiResponse(BaseModel):
    words: list[str]
    speakers: list[str]
    start: list[float]
    end: list[float]

    @property
    def transcript(self) -> str:
        return " ".join(self.words)

    @model_validator(mode="after")
    def validate_lengths(self) -> "DeepgramApiResponse":
        if (
            len(self.words) != len(self.speakers)
            or len(self.words) != len(self.start)
            or len(self.words) != len(self.end)
        ):
            raise ValueError("All lists must be of the same length")
        return self


class DeepgramApi:
    def __init__(self, options: PrerecordedOptions):
        self.options = options

        # Check that the API key is set
        if not os.getenv("DEEPGRAM_API_KEY"):
            raise ValueError("`DEEPGRAM_API_KEY` is not set")

        self.client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

    # Only intended to be used with offiline transcription
    def transcribe(self, audio_path: Path | str) -> DeepgramApiResponse:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        with audio_path.open("rb") as file:
            buffer_data = file.read()
        payload: FileSource = {"buffer": buffer_data}

        response: PrerecordedResponse = self.client.listen.rest.v("1").transcribe_file(payload, self.options)

        return DeepgramApiResponse(
            words=[w.punctuated_word for w in response.results.channels[0].alternatives[0].words],
            speakers=[str(w.speaker) for w in response.results.channels[0].alternatives[0].words],
            start=[float(w.start) for w in response.results.channels[0].alternatives[0].words],
            end=[float(w.end) for w in response.results.channels[0].alternatives[0].words],
        )
