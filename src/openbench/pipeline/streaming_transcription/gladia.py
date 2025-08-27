# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import asyncio
import json
import os
import time
from typing import Literal, TypedDict

import numpy as np
import requests
from argmaxtools.utils import get_logger
from websockets.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosedOK

from openbench.dataset import StreamingSample

from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import StreamingTranscript
from ...types import PipelineType
from .common import StreamingTranscriptionConfig, StreamingTranscriptionOutput


logger = get_logger(__name__)

# Some parts of this code are adapted from the example provided at:
# https://github.com/gladiaio/gladia-samples/tree/main/python/src/streaming


class GladiaApi:
    def __init__(self, cfg) -> None:
        self.chunk_size_ms = cfg.chunksize_ms
        self.api_key = os.getenv("GLADIA_API_KEY")
        assert self.api_key is not None, "Please set API key in environment"
        self.channels = cfg.channels
        self.sample_width = cfg.sample_width
        self.sample_rate = cfg.sample_rate
        self.api_endpoint_base_url = cfg.endpoint_url

    def run(self, data):
        global confirmed_interim_transcripts
        global audio_cursor
        global confirmed_audio_cursor_l
        global predicted_transcript_hypot
        predicted_transcript_hypot = ""
        confirmed_audio_cursor_l = []
        audio_cursor = 0
        confirmed_interim_transcripts = []

        class InitiateResponse(TypedDict):
            id: str
            url: str

        class LanguageConfiguration(TypedDict):
            languages: list[str] | None
            code_switching: bool | None

        class StreamingConfiguration(TypedDict):
            # This is a reduced set of options. For a full list, see the API documentation.
            # https://docs.gladia.io/api-reference/v2/live/init
            encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"]
            bit_depth: Literal[8, 16, 24, 32]
            sample_rate: Literal[8_000, 16_000, 32_000, 44_100, 48_000]
            channels: int
            language_config: LanguageConfiguration | None

        def init_live_session(config: StreamingConfiguration) -> InitiateResponse:
            response = requests.post(
                self.api_endpoint_base_url,
                headers={"X-Gladia-Key": self.api_key},
                json=config,
                timeout=3,
            )
            if not response.ok:
                print(f"{response.status_code}: {response.text or response.reason}")
                exit(response.status_code)
            return response.json()

        def format_duration(seconds: float) -> str:
            milliseconds = int(seconds * 1_000)
            return time(
                hour=milliseconds // 3_600_000,
                minute=(milliseconds // 60_000) % 60,
                second=(milliseconds // 1_000) % 60,
                microsecond=milliseconds % 1_000 * 1_000,
            ).isoformat(timespec="milliseconds")

        async def print_messages_from_socket(socket: ClientConnection):
            global final_transcript
            global audio_cursor
            global confirmed_audio_cursor_l
            global model_timestamps_confirmed
            model_timestamps_confirmed = []
            final_transcript = ""
            async for message in socket:
                content = json.loads(message)
                if content["type"] == "transcript":
                    confirmed_audio_cursor_l.append(audio_cursor)
                    text = content["data"]["utterance"]["text"].strip()
                    confirmed_interim_transcripts.append(text)
                    model_timestamps_confirmed.append(content["data"]["utterance"]["words"])
                    logger.debug("\n" + "Transcription: " + content["data"]["utterance"]["text"].strip())
                if content["type"] == "post_final_transcript":
                    logger.debug("\n################ End of session ################\n")
                    final_transcript = content["data"]["transcription"]["full_transcript"]

        async def stop_recording(websocket: ClientConnection) -> None:
            logger.debug(">>>>> Ending the recording…")
            await websocket.send(json.dumps({"type": "stop_recording"}))
            await asyncio.sleep(0)

        STREAMING_CONFIGURATION: StreamingConfiguration = {
            # This configuration is for a 16kHz, 16-bit, mono PCM WAV file
            "encoding": "wav/pcm",
            "sample_rate": 16_000,
            "bit_depth": 16,
            "channels": 1,
            "language_config": {
                "languages": ["es", "ru", "en", "fr"],
                "code_switching": True,
            },
        }

        async def send_audio(socket: ClientConnection, data) -> None:
            global audio_cursor
            audio_duration_in_seconds = 0.1
            chunk_size = int(
                STREAMING_CONFIGURATION["sample_rate"]
                * (STREAMING_CONFIGURATION["bit_depth"] / 8)
                * STREAMING_CONFIGURATION["channels"]
                * audio_duration_in_seconds
            )

            # Send the audio file in chunks
            offset = 0
            while offset < len(data):
                try:
                    await socket.send(data[offset : offset + chunk_size])
                    offset += chunk_size
                    audio_cursor += audio_duration_in_seconds
                    await asyncio.sleep(audio_duration_in_seconds)
                except ConnectionClosedOK:
                    return
            logger.debug(">>>>> Sent all audio data")
            await stop_recording(socket)

        async def main(data):
            response = init_live_session(STREAMING_CONFIGURATION)
            async with connect(response["url"]) as websocket:
                try:
                    logger.debug("\n################ Begin session ################\n")
                    tasks = []
                    tasks.append(asyncio.create_task(send_audio(websocket, data)))
                    tasks.append(asyncio.create_task(print_messages_from_socket(websocket)))

                    await asyncio.wait(tasks)
                except asyncio.exceptions.CancelledError:
                    for task in tasks:
                        task.cancel()
                    await stop_recording(websocket)
                    await print_messages_from_socket(websocket)

        asyncio.run(main(data))

        return (
            final_transcript,
            None,
            None,  # no hypothesis text is provided by Gladia
            confirmed_interim_transcripts,
            confirmed_audio_cursor_l,
            None,
            model_timestamps_confirmed,
        )

    def __call__(self, sample):
        # Sample must be in bytes
        (
            transcript,
            interim_transcripts,
            audio_cursor_l,
            confirmed_interim_transcripts,
            confirmed_audio_cursor_l,
            model_timestamps_hypothesis,
            model_timestamps_confirmed,
        ) = self.run(sample)
        return {
            "transcript": transcript,
            "interim_transcripts": interim_transcripts,  # no hypothesis text is provided by Gladia
            "audio_cursor": audio_cursor_l,
            "confirmed_interim_transcripts": confirmed_interim_transcripts,
            "confirmed_audio_cursor": confirmed_audio_cursor_l,
            "model_timestamps_hypothesis": model_timestamps_hypothesis,
            "model_timestamps_confirmed": model_timestamps_confirmed,
        }


class GladiaStreamingPipelineConfig(StreamingTranscriptionConfig):
    sample_rate: int
    channels: int
    sample_width: int
    chunksize_ms: float


@register_pipeline
class GladiaStreamingPipeline(Pipeline):
    _config_class = GladiaStreamingPipelineConfig
    pipeline_type = PipelineType.STREAMING_TRANSCRIPTION

    def parse_input(self, input_sample: StreamingSample):
        y = input_sample.waveform
        y_int16 = (y * 32767).astype(np.int16)
        audio_data_byte = y_int16.T.tobytes()
        return audio_data_byte

    def parse_output(self, output) -> StreamingTranscriptionOutput:
        model_timestamps_hypothesis = output["model_timestamps_hypothesis"]
        model_timestamps_confirmed = output["model_timestamps_confirmed"]

        if model_timestamps_hypothesis is not None:
            model_timestamps_hypothesis = [
                [{"start": word["start"], "end": word["end"]} for word in interim_result_words]
                for interim_result_words in model_timestamps_hypothesis
            ]

        if model_timestamps_confirmed is not None:
            model_timestamps_confirmed = [
                [{"start": word["start"], "end": word["end"]} for word in interim_result_words]
                for interim_result_words in model_timestamps_confirmed
            ]

        prediction = StreamingTranscript(
            transcript=output["transcript"],
            audio_cursor=output["audio_cursor"],
            interim_results=output["interim_transcripts"],
            confirmed_audio_cursor=output["confirmed_audio_cursor"],
            confirmed_interim_results=output["confirmed_interim_transcripts"],
            model_timestamps_hypothesis=model_timestamps_hypothesis,
            model_timestamps_confirmed=model_timestamps_confirmed,
        )

        return StreamingTranscriptionOutput(prediction=prediction)

    def build_pipeline(self):
        pipeline = GladiaApi(self.config)
        return pipeline
