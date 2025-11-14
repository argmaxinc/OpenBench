# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import asyncio
import json
import os

import numpy as np
import websockets
from argmaxtools.utils import get_logger
from pyannote.core import Segment
from pydantic import Field

from openbench.dataset import OrchestrationSample

from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import (
    DiarizationAnnotation,
    StreamingDiarization,
    Transcript,
    Word,
)
from ...types import PipelineType
from .common import StreamingDiarizationConfig, StreamingDiarizationOutput


logger = get_logger(__name__)

# Some parts of this code are adapted from the Deepgram streaming example at:
# https://developers.deepgram.com/docs/measuring-streaming-latency


class DeepgramStreamingDiarizationApi:
    def __init__(self, cfg) -> None:
        self.realtime_resolution = 0.020
        self.model_version = "nova-3"
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        assert (
            self.api_key is not None
        ), "Please set DEEPGRAM_API_KEY in environment"
        self.channels = cfg.channels
        self.sample_width = cfg.sample_width
        self.sample_rate = cfg.sample_rate
        self.host_url = os.getenv(
            "DEEPGRAM_HOST_URL", "wss://api.deepgram.com"
        )

    async def run(self, data, key, channels, sample_width, sample_rate):
        """Connect to Deepgram real-time endpoint with diarization.

        This streams audio data in real-time and collects diarization results.
        """
        # How many bytes are contained in one second of audio.
        byte_rate = sample_width * sample_rate * channels

        # Variables for collecting results
        audio_cursor = 0.0
        audio_cursor_l = []
        interim_annotations_l = []
        confirmed_audio_cursor_l = []
        confirmed_interim_annotations_l = []
        final_annotation = DiarizationAnnotation()
        transcript_text = ""
        words_with_speakers = []

        # Connect to the real-time streaming endpoint with diarization
        url = (
            f"{self.host_url}/v1/listen?"
            f"model={self.model_version}&"
            f"channels={channels}&"
            f"sample_rate={sample_rate}&"
            f"encoding=linear16&"
            f"interim_results=true&"
            f"diarize=true"
        )
        async with websockets.connect(
            url,
            additional_headers={
                "Authorization": "Token {}".format(key),
            },
        ) as ws:

            async def sender(ws):
                """Sends the data, mimicking real-time connection."""
                nonlocal data, audio_cursor
                try:
                    while len(data):
                        # Bytes in `REALTIME_RESOLUTION` seconds
                        i = int(byte_rate * self.realtime_resolution)
                        chunk, data = data[:i], data[i:]
                        # Send the data
                        await ws.send(chunk)
                        # Move the audio cursor
                        audio_cursor += self.realtime_resolution
                        # Mimic real-time by waiting
                        await asyncio.sleep(self.realtime_resolution)

                    # A CloseStream message tells Deepgram that no more audio
                    # will be sent. Deepgram will close the connection once all
                    # audio has finished processing.
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except Exception as e:
                    logger.error(f"Error while sending: {e}")
                    raise

            async def receiver(ws):
                """Collect diarization results from the server."""
                nonlocal audio_cursor
                nonlocal interim_annotations_l
                nonlocal audio_cursor_l
                nonlocal confirmed_interim_annotations_l
                nonlocal confirmed_audio_cursor_l
                nonlocal final_annotation
                nonlocal transcript_text
                nonlocal words_with_speakers

                async for msg in ws:
                    msg = json.loads(msg)

                    if "request_id" in msg:
                        # This is the final metadata message
                        continue

                    # Process words with speaker information
                    if "channel" in msg and "alternatives" in msg["channel"]:
                        alternatives = msg["channel"]["alternatives"]
                        if (
                            len(alternatives) > 0
                            and "words" in alternatives[0]
                        ):
                            words = alternatives[0]["words"]

                            # Create annotation from words
                            annotation = DiarizationAnnotation()
                            for word_info in words:
                                if (
                                    "speaker" in word_info
                                    and "start" in word_info
                                    and "end" in word_info
                                ):
                                    speaker = word_info["speaker"]
                                    start = word_info["start"]
                                    end = word_info["end"]
                                    segment = Segment(start, end)
                                    annotation[segment] = (
                                        f"SPEAKER_{speaker}"
                                    )

                            if len(annotation) > 0:
                                if not msg.get("is_final", False):
                                    # Interim result
                                    audio_cursor_l.append(audio_cursor)
                                    interim_annotations_l.append(
                                        annotation
                                    )
                                    logger.debug(
                                        f"Interim annotation with "
                                        f"{len(annotation)} segments"
                                    )
                                else:
                                    # Confirmed/final result
                                    confirmed_audio_cursor_l.append(
                                        audio_cursor
                                    )
                                    confirmed_interim_annotations_l.append(
                                        annotation
                                    )

                                    # Merge into final annotation
                                    for (
                                        segment,
                                        _,
                                        speaker,
                                    ) in annotation.itertracks(
                                        yield_label=True
                                    ):
                                        final_annotation[segment] = speaker

                                    # Collect final transcript and words
                                    for word_info in words:
                                        if (
                                            "word" in word_info
                                            and "speaker" in word_info
                                        ):
                                            speaker_label = (
                                                f"SPEAKER_"
                                                f"{word_info['speaker']}"
                                            )
                                            words_with_speakers.append({
                                                "word": word_info["word"],
                                                "speaker": speaker_label,
                                                "start": (
                                                    word_info.get("start", 0)
                                                ),
                                                "end": (
                                                    word_info.get("end", 0)
                                                ),
                                            })

                                    # Build full transcript with tags
                                    if words_with_speakers:
                                        current_speaker = None
                                        transcript_parts = []
                                        for w in words_with_speakers:
                                            spk = w["speaker"]
                                            if spk != current_speaker:
                                                if (
                                                    current_speaker
                                                    is not None
                                                ):
                                                    transcript_parts.append(
                                                        ""
                                                    )
                                                transcript_parts.append(
                                                    f"[{spk}]"
                                                )
                                                current_speaker = spk
                                            transcript_parts.append(
                                                w["word"]
                                            )
                                        transcript_text = " ".join(
                                            transcript_parts
                                        )

                                    logger.debug(
                                        f"Confirmed annotation with "
                                        f"{len(annotation)} segments"
                                    )

            await asyncio.wait([
                asyncio.ensure_future(sender(ws)),
                asyncio.ensure_future(receiver(ws))
            ])

            return (
                final_annotation,
                interim_annotations_l,
                audio_cursor_l,
                confirmed_interim_annotations_l,
                confirmed_audio_cursor_l,
                transcript_text,
                words_with_speakers,
            )

    def __call__(self, sample):
        # Sample must be in bytes
        (
            final_annotation,
            interim_annotations,
            audio_cursor_l,
            confirmed_interim_annotations,
            confirmed_audio_cursor_l,
            transcript_text,
            words_with_speakers,
        ) = asyncio.get_event_loop().run_until_complete(
            self.run(
                sample,
                self.api_key,
                self.channels,
                self.sample_width,
                self.sample_rate,
            )
        )

        return {
            "annotation": final_annotation,
            "interim_annotations": interim_annotations,
            "audio_cursor": audio_cursor_l,
            "confirmed_interim_annotations": confirmed_interim_annotations,
            "confirmed_audio_cursor": confirmed_audio_cursor_l,
            "transcript_text": transcript_text,
            "words": words_with_speakers,
        }


class DeepgramStreamingDiarizationPipelineConfig(StreamingDiarizationConfig):
    sample_rate: int
    channels: int
    sample_width: int
    realtime_resolution: float
    model_version: str = Field(
        default="nova-3",
        description="The model to use for real-time diarization"
    )


@register_pipeline
class DeepgramStreamingDiarizationPipeline(Pipeline):
    _config_class = DeepgramStreamingDiarizationPipelineConfig
    pipeline_type = PipelineType.STREAMING_DIARIZATION

    def parse_input(self, input_sample: OrchestrationSample):
        y = input_sample.waveform
        y_int16 = (y * 32767).astype(np.int16)
        audio_data_byte = y_int16.T.tobytes()
        return audio_data_byte

    def parse_output(self, output) -> StreamingDiarizationOutput:
        # Create Transcript from words with speakers
        # For cpWER/WDER, we return transcript as the main prediction
        words = [
            Word(
                word=w["word"],
                start=w.get("start"),
                end=w.get("end"),
                speaker=w.get("speaker"),
            )
            for w in output["words"]
        ]
        transcript = Transcript(words=words)

        return StreamingDiarizationOutput(prediction=transcript)

    def build_pipeline(self):
        pipeline = DeepgramStreamingDiarizationApi(self.config)
        return pipeline
