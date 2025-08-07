# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import asyncio
import json
import os

import numpy as np
import websockets
from argmaxtools.utils import get_logger
from pydantic import Field

from openbench.dataset import StreamingSample

from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import StreamingTranscript
from ...types import PipelineType
from .common import StreamingTranscriptionConfig, StreamingTranscriptionOutput


logger = get_logger(__name__)

# Some parts of this code are adapted from the example provided at:
# https://developers.deepgram.com/docs/measuring-streaming-latency


class DeepgramApi:
    def __init__(self, cfg) -> None:
        self.realtime_resolution = 0.020
        self.model_version = "nova-3"
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        assert self.api_key is not None, "Please set API key in environment"
        self.channels = cfg.channels
        self.sample_width = cfg.sample_width
        self.sample_rate = cfg.sample_rate
        self.host_url = os.getenv("DEEPGRAM_HOST_URL", "wss://api.deepgram.com")

    async def run(self, data, key, channels, sample_width, sample_rate):
        """Connect to the Deepgram real-time streaming endpoint, stream the data
        in real-time, and print out the responses from the server.

        This uses a pre-recorded file as an example. It mimics a real-time
        connection by sending `REALTIME_RESOLUTION` seconds of audio every
        `REALTIME_RESOLUTION` seconds of wall-clock time.

        This is a toy example, since it uses a pre-recorded file. In a real use
        case, you'd be streaming the audio stream itself. If you actually have
        pre-recorded audio, use Deepgram's pre-recorded mode instead.
        """
        # How many bytes are contained in one second of audio.
        byte_rate = sample_width * sample_rate * channels
        global audio_cursor_l
        global interim_transcripts
        global confirmed_audio_cursor_l
        global confirmed_interim_transcripts
        global model_timestamps_hypothesis
        global model_timestamps_confirmed
        audio_cursor = 0.0
        audio_cursor_l = []
        interim_transcripts = []
        confirmed_audio_cursor_l = []
        confirmed_interim_transcripts = []
        model_timestamps_hypothesis = []
        model_timestamps_confirmed = []
        # Connect to the real-time streaming endpoint, attaching our API key.
        async with websockets.connect(
            f"{self.host_url}/v1/listen?model={self.model_version}&channels={channels}&sample_rate={sample_rate}&encoding=linear16&interim_results=true",
            additional_headers={
                "Authorization": "Token {}".format(key),
            },
        ) as ws:

            async def sender(ws):
                """Sends the data, mimicking a real-time connection."""
                nonlocal data, audio_cursor
                try:
                    while len(data):
                        # How many bytes are in `REALTIME_RESOLUTION` seconds of audio
                        i = int(byte_rate * self.realtime_resolution)
                        chunk, data = data[:i], data[i:]
                        # Send the data
                        await ws.send(chunk)
                        # Move the audio cursor
                        audio_cursor += self.realtime_resolution
                        # Mimic real-time by waiting `REALTIME_RESOLUTION` seconds
                        # before the next packet.
                        await asyncio.sleep(self.realtime_resolution)

                    # A CloseStream message tells Deepgram that no more audio
                    # will be sent. Deepgram will close the connection once all
                    # audio has finished processing.
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except Exception as e:
                    print(f"Error while sending: {e}")
                    raise

            async def receiver(ws):
                """Print out the messages received from the server."""
                nonlocal audio_cursor
                global transcript
                global interim_transcripts
                global audio_cursor_l
                global confirmed_interim_transcripts
                global confirmed_audio_cursor_l
                global model_timestamps_hypothesis
                global model_timestamps_confirmed
                transcript = ""

                async for msg in ws:
                    msg = json.loads(msg)
                    if "request_id" in msg:
                        # This is the final metadata message. It gets sent as the
                        # very last message by Deepgram during a clean shutdown.
                        # There is no transcript in it.
                        continue
                    if msg["channel"]["alternatives"][0]["transcript"] != "":
                        if not msg["is_final"]:
                            audio_cursor_l.append(audio_cursor)
                            model_timestamps_hypothesis.append(msg["channel"]["alternatives"][0]["words"])
                            interim_transcripts.append(
                                transcript + " " + msg["channel"]["alternatives"][0]["transcript"]
                            )
                            logger.debug(
                                "\n" + "Transcription: " + transcript + msg["channel"]["alternatives"][0]["transcript"]
                            )

                        elif msg["is_final"] and (not msg["from_finalize"]):
                            confirmed_audio_cursor_l.append(audio_cursor)
                            transcript = transcript + " " + msg["channel"]["alternatives"][0]["transcript"]
                            confirmed_interim_transcripts.append(transcript)
                            model_timestamps_confirmed.append(msg["channel"]["alternatives"][0]["words"])

                        elif msg["is_final"] and msg["from_finalize"]:
                            confirmed_audio_cursor_l.append(audio_cursor)
                            transcript = msg["channel"]["alternatives"][0]["transcript"]
                            confirmed_interim_transcripts.append(transcript)
                            model_timestamps_confirmed.append(msg["channel"]["alternatives"][0]["words"])

            await asyncio.wait([asyncio.ensure_future(sender(ws)), asyncio.ensure_future(receiver(ws))])
            return (
                transcript,
                interim_transcripts,
                audio_cursor_l,
                confirmed_interim_transcripts,
                confirmed_audio_cursor_l,
                model_timestamps_hypothesis,
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
        ) = asyncio.get_event_loop().run_until_complete(
            self.run(sample, self.api_key, self.channels, self.sample_width, self.sample_rate)
        )
        return {
            "transcript": transcript,
            "interim_transcripts": interim_transcripts,
            "audio_cursor": audio_cursor_l,
            "confirmed_interim_transcripts": confirmed_interim_transcripts,
            "confirmed_audio_cursor": confirmed_audio_cursor_l,
            "model_timestamps_hypothesis": model_timestamps_hypothesis,
            "model_timestamps_confirmed": model_timestamps_confirmed,
        }


class DeepgramStreamingPipelineConfig(StreamingTranscriptionConfig):
    sample_rate: int
    channels: int
    sample_width: int
    realtime_resolution: float
    model_version: str = Field(..., description="The model to use for real-time transcription")


@register_pipeline
class DeepgramStreamingPipeline(Pipeline):
    _config_class = DeepgramStreamingPipelineConfig
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
        pipeline = DeepgramApi(self.config)
        return pipeline
