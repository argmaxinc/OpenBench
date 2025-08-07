# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import asyncio
import base64
import json
import os

import aiohttp
import numpy as np
import torch
import torchaudio
import websockets
from argmaxtools.utils import get_logger

from openbench.dataset import StreamingSample

from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import StreamingTranscript
from ...types import PipelineType
from .common import StreamingTranscriptionConfig, StreamingTranscriptionOutput


logger = get_logger(__name__)

# Some parts of this code are adapted from the example provided at:
# https://community.openai.com/t/use-new-model-for-realtime-audio-transcription/1154610


class OpenAIApi:
    def __init__(self, cfg) -> None:
        self.realtime_resolution = cfg.realtime_resolution
        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "Please set API key in environment"
        self.channels = cfg.channels
        self.sample_width = cfg.sample_width
        self.sample_rate = cfg.sample_rate
        self.api_endpoint_base_url = cfg.endpoint_url
        self.model_version = cfg.model

    def run(self, data):
        global confirmed_interim_transcripts
        global audio_cursor
        global confirmed_audio_cursor_l
        global final_transcription
        final_transcription = ""
        confirmed_audio_cursor_l = []
        audio_cursor = 0
        confirmed_interim_transcripts = []

        async def create_transcription_session():
            """
            Create a transcription session via the REST API to obtain an ephemeral token.
            This endpoint uses the beta header "OpenAI-Beta: assistants=v2".
            """
            url = "https://api.openai.com/v1/realtime/transcription_sessions"
            payload = {
                "input_audio_transcription": {
                    "model": self.model_version,
                    "language": "en",
                    # "prompt": "Transcribe the incoming audio in real time."
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                # "turn_detection": None
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "assistants=v2",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(f"Failed to create transcription session: {resp.status} {text}")
                    data = await resp.json()
                    ephemeral_token = data["client_secret"]["value"]
                    logger.debug("Transcription session created; ephemeral token obtained.")
                    return ephemeral_token

        async def send_audio(
            ws,
            data,
            byte_rate: int,
            speech_stopped_event: asyncio.Event,
        ):
            """
            Read the local ulaw file and send it in chunks.
            After finishing, wait for 1 second to see if the server auto-commits.
            If not, send a commit event manually.
            """
            global audio_finished
            global audio_cursor
            try:
                while len(data):
                    i = int(byte_rate * self.realtime_resolution)
                    chunk, data = data[:i], data[i:]
                    audio_chunk = base64.b64encode(chunk).decode("utf-8")
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_chunk,
                    }

                    await ws.send(json.dumps(audio_event))
                    await asyncio.sleep(self.realtime_resolution)  # simulate real-time streaming
                    audio_cursor += self.realtime_resolution
                # Wait 1 second to allow any late VAD events before committing.
                try:
                    await asyncio.wait_for(speech_stopped_event.wait(), timeout=0.5)
                    logger.debug("Speech stopped event received; no manual commit needed.")
                except asyncio.TimeoutError:
                    commit_event = {"type": "input_audio_buffer.commit"}
                    await ws.send(json.dumps(commit_event))
                    audio_finished = True
                    logger.debug("Manually sent input_audio_buffer.commit event.")
            except Exception as e:
                logger.error("Error sending audio: %s", e)

        async def receive_events(ws, speech_stopped_event: asyncio.Event):
            """
            Listen for events from the realtime endpoint.
            Capture transcription deltas and the final complete transcription.
            Set the speech_stopped_event when a "speech_stopped" event is received.
            """
            global final_transcription
            global confirmed_audio_cursor_l
            global confirmed_interim_transcripts
            try:
                async for message in ws:
                    try:
                        event = json.loads(message)
                        event_type = event.get("type")
                        if event_type == "conversation.item.input_audio_transcription.delta":
                            confirmed_audio_cursor_l.append(audio_cursor)
                            delta = event.get("delta", "")
                            logger.debug("Transcription delta: %s", delta)
                            final_transcription += delta
                            confirmed_interim_transcripts.append(final_transcription)
                            logger.debug("\n" + "Transcription: " + final_transcription)
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            final_transcription = final_transcription + " "
                            if audio_finished:
                                break  # Exit after final transcription
                        elif event_type == "error":
                            logger.error("Error event: %s", event.get("error"))
                        else:
                            logger.debug("Received event: %s", event_type)
                    except Exception as ex:
                        logger.error("Error processing message: %s", ex)
            except Exception as e:
                logger.error("Error receiving events: %s", e)

        async def test_transcription(data):
            global final_transcription
            try:
                # Step 1: Create transcription session and get ephemeral token.
                ephemeral_token = await create_transcription_session()
                # Step 2: Connect to the base realtime endpoint.
                websocket_url = "wss://api.openai.com/v1/realtime"
                connection_headers = {
                    "Authorization": f"Bearer {ephemeral_token}",
                    "OpenAI-Beta": "realtime=v1",
                }
                async with websockets.connect(websocket_url, extra_headers=connection_headers) as ws:
                    logger.debug("Connected to realtime endpoint.")

                    # Step 3: Send transcription session update event with adjusted VAD settings.
                    update_event = {
                        "type": "transcription_session.update",
                        "session": {
                            "input_audio_transcription": {
                                "model": self.model_version,
                                "language": "en",
                                # "prompt": "Transcribe the incoming audio in real time."
                            },
                            # "turn_detection": None
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500,
                            },
                        },
                    }
                    await ws.send(json.dumps(update_event))
                    logger.debug("Sent transcription session update event.")

                    # Create an event to signal if speech stopped is detected.
                    speech_stopped_event = asyncio.Event()

                    # Step 4: Run sender and receiver concurrently.
                    global audio_finished
                    audio_finished = False

                    byte_rate = self.sample_width * self.sample_rate * self.channels

                    sender_task = asyncio.create_task(send_audio(ws, data, byte_rate, speech_stopped_event))
                    receiver_task = asyncio.create_task(receive_events(ws, speech_stopped_event))

                    await asyncio.wait_for(
                        asyncio.gather(sender_task, receiver_task),
                        timeout=2 * len(data) / byte_rate,  # timeout in seconds,
                        # 2* audio duration
                    )

            except Exception as e:
                logger.error("Error in transcription test: %s", e)

        asyncio.run(test_transcription(data))

        return (
            final_transcription,
            None,
            None,
            confirmed_interim_transcripts,
            confirmed_audio_cursor_l,
            None,
            None,
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


class OpenAIStreamingPipelineConfig(StreamingTranscriptionConfig):
    sample_rate: int
    channels: int
    sample_width: int
    realtime_resolution: float
    model: str


@register_pipeline
class OpenAIStreamingPipeline(Pipeline):
    _config_class = OpenAIStreamingPipelineConfig
    pipeline_type = PipelineType.STREAMING_TRANSCRIPTION

    def audio2chunks(self, audio_data):
        audio_data = audio_data[None, :]
        # Resample to 16000 Hz
        target_sample_rate = 16000
        audio_data = torchaudio.functional.resample(
            torch.Tensor(audio_data), self.config.sample_rate, target_sample_rate
        )
        print(
            f"Resampled audio tensor. shape={audio_data.shape} \
            sample_rate={target_sample_rate}"
        )

        # Convert to mono
        audio_tensor = torch.Tensor(audio_data).mean(dim=0, keepdim=True)
        print(f"Mono audio tensor. shape={audio_tensor.shape}")

        audio_chunk_tensors = torch.split(
            audio_tensor,
            int(self.config.chunksize_ms * target_sample_rate / 1000),
            dim=1,
        )
        print(
            f"Split into {len(audio_chunk_tensors)} audio \
            chunks each {self.config.chunksize_ms}ms"
        )

        audio_chunk_bytes = []
        for audio_chunk_tensor in audio_chunk_tensors:
            audio_chunk_bytes.append((audio_chunk_tensor * 32768.0).to(torch.int16).numpy().tobytes())

        return audio_chunk_bytes

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
        pipeline = OpenAIApi(self.config)
        return pipeline
