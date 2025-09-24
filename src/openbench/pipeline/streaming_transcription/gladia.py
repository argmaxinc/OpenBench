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
        global interim_transcripts_list
        global audio_cursor_l
        global model_timestamps_hypothesis
        global cumulative_transcript
        predicted_transcript_hypot = ""
        confirmed_audio_cursor_l = []
        interim_transcripts_list = []
        audio_cursor_l = []
        model_timestamps_hypothesis = []
        audio_cursor = 0
        confirmed_interim_transcripts = []
        cumulative_transcript = ""

        class InitiateResponse(TypedDict):
            id: str
            url: str

        class LanguageConfiguration(TypedDict):
            languages: list[str] | None
            code_switching: bool | None

        class StreamingConfiguration(TypedDict):
            # This is a reduced set of options. For a full list, see the API
            # documentation.
            # https://docs.gladia.io/api-reference/v2/live/init
            encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"]
            bit_depth: Literal[8, 16, 24, 32]
            sample_rate: Literal[
                8_000, 16_000, 32_000, 44_100, 48_000
            ]
            channels: int
            language_config: LanguageConfiguration | None

        def init_live_session(config: StreamingConfiguration) -> InitiateResponse:
            response = requests.post(
                self.api_endpoint_base_url,
                headers={
                    "Content-Type": "application/json",
                    "X-Gladia-Key": self.api_key
                },
                json=config,
                timeout=10,
            )
            if not response.ok:
                error_text = response.text or response.reason
                error_message = f"{response.status_code}: {error_text}"
                logger.error(
                    f"Failed to initialize Gladia live session: {error_message}"
                )
                raise RuntimeError(
                    f"Gladia API initialization failed: {error_message}"
                )
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
            global predicted_transcript_hypot
            global interim_transcripts_list
            global audio_cursor_l
            global model_timestamps_hypothesis
            global cumulative_transcript
            model_timestamps_confirmed = []
            final_transcript = ""

            try:
                async for message in socket:
                    try:
                        content = json.loads(message)
                        message_type = content.get("type")
                        if message_type == "transcript":
                            data = content.get("data", {})
                            utterance = data.get("utterance", {})
                            text = utterance.get("text", "").strip()
                            words = utterance.get("words", [])
                            is_final = data.get("is_final", True)
                            if text:
                                if not is_final:
                                    # Only track interim transcripts with valid timestamps
                                    if (words and len(words) > 0 and 
                                        all("start" in word and "end" in word 
                                            for word in words)):
                                        cumulative_with_new = (
                                            cumulative_transcript + " " + text
                                            if cumulative_transcript else text
                                        ).strip()
                                        interim_transcripts_list.append(
                                            cumulative_with_new
                                        )
                                        audio_cursor_l.append(audio_cursor)
                                        model_timestamps_hypothesis.append(words)
                                        predicted_transcript_hypot = text
                                        logger.debug(f"Interim transcript: {text}")
                                else:
                                    confirmed_audio_cursor_l.append(
                                        audio_cursor
                                    )
                                    # Update cumulative transcript
                                    cumulative_transcript = (
                                        cumulative_transcript + " " + text
                                        if cumulative_transcript else text
                                    ).strip()
                                    confirmed_interim_transcripts.append(
                                        cumulative_transcript
                                    )
                                    model_timestamps_confirmed.append(words)
                                    logger.debug(f"Final transcript: {text}")
                        elif message_type == "post_final_transcript":
                            logger.debug(
                                "\n#### End of Gladia v2 session ####\n"
                            )
                            transcription_data = content.get(
                                "data", {}
                            ).get("transcription", {})
                            final_transcript = transcription_data.get(
                                "full_transcript", ""
                            )
                        else:
                            logger.debug(
                                f"Received message type: {message_type}"
                            )
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(
                            f"Failed to parse message from Gladia: {e}"
                        )
                        continue
            except Exception as e:
                logger.error(
                    f"Error reading messages from Gladia websocket: {e}"
                )
                raise

        async def stop_recording(websocket: ClientConnection) -> None:
            await websocket.send(json.dumps({"type": "stop_recording"}))
            await asyncio.sleep(0)

        STREAMING_CONFIGURATION: StreamingConfiguration = {
            "encoding": "wav/pcm",
            "sample_rate": self.sample_rate,
            "bit_depth": self.sample_width * 8,
            "channels": self.channels,
            "language_config": {
                "languages": ["es", "ru", "en", "fr"],
                "code_switching": True,
            },
            "messages_config": {
                                "receive_partial_transcripts": True,
                                "receive_final_transcripts": True,
                                "receive_speech_events": True,
                                "receive_pre_processing_events": False,
                                "receive_realtime_processing_events": True,
                                "receive_post_processing_events": False,
                                "receive_acknowledgments": False,
                                "receive_lifecycle_events": False                    
                                }
        }

        async def send_audio(socket: ClientConnection, data) -> None:
            global audio_cursor
            audio_duration_in_seconds = self.chunk_size_ms / 1000.0
            chunk_size = int(
                STREAMING_CONFIGURATION["sample_rate"]
                * (STREAMING_CONFIGURATION["bit_depth"] / 8)
                * STREAMING_CONFIGURATION["channels"]
                * audio_duration_in_seconds
            )

            offset = 0
            while offset < len(data):
                try:
                    actual_chunk_size = min(chunk_size, len(data) - offset)
                    await socket.send(data[offset: offset + actual_chunk_size])
                    offset += actual_chunk_size
                    actual_audio_duration = actual_chunk_size / (
                        STREAMING_CONFIGURATION["sample_rate"]
                        * (STREAMING_CONFIGURATION["bit_depth"] / 8)
                        * STREAMING_CONFIGURATION["channels"]
                    )
                    audio_cursor += actual_audio_duration
                    await asyncio.sleep(audio_duration_in_seconds)
                except ConnectionClosedOK:
                    return
            logger.debug(">>>>> Sent all audio data")
            await stop_recording(socket)

        async def main(data):
            try:
                response = init_live_session(STREAMING_CONFIGURATION)
                websocket_url = response["url"]
                logger.debug(f"Received websocket URL from Gladia API: {websocket_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Gladia session: {e}")
                raise
            async with connect(websocket_url) as websocket:
                try:
                    logger.debug(
                        "\n#### Begin Gladia session ####\n"
                    )
                    tasks = []
                    tasks.append(
                        asyncio.create_task(send_audio(websocket, data))
                    )
                    tasks.append(
                        asyncio.create_task(
                            print_messages_from_socket(websocket)
                        )
                    )

                    await asyncio.wait(tasks)
                except asyncio.exceptions.CancelledError:
                    logger.debug("Tasks cancelled, cleaning up...")
                    for task in tasks:
                        task.cancel()
                    await stop_recording(websocket)
                    await print_messages_from_socket(websocket)
                except Exception as e:
                    logger.error(
                        f"Error during websocket communication: {e}"
                    )
                    raise

        asyncio.run(main(data))

        return (
            final_transcript if final_transcript else cumulative_transcript,
            (interim_transcripts_list
             if interim_transcripts_list else None),
            (audio_cursor_l
             if audio_cursor_l else None),
            confirmed_interim_transcripts,
            confirmed_audio_cursor_l,
            (model_timestamps_hypothesis
             if model_timestamps_hypothesis else None),
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
            "interim_transcripts": interim_transcripts,
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
