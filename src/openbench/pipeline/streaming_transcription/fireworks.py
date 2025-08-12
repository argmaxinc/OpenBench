# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
import os
import threading
import time
import urllib.parse

import torch
import torchaudio
import websocket
from argmaxtools.utils import get_logger

from openbench.dataset import StreamingSample

from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import StreamingTranscript
from ...types import PipelineType
from .common import StreamingTranscriptionConfig, StreamingTranscriptionOutput


logger = get_logger(__name__)

# Some parts of this code are adapted from the example provided at:
# https://docs.fireworks.ai/api-reference/audio-streaming-transcriptions


class FireworksApi:
    def __init__(self, cfg) -> None:
        self.chunk_size_ms = cfg.chunksize_ms
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        assert self.api_key is not None, "Please set API key in environment"
        self.channels = cfg.channels
        self.sample_width = cfg.sample_width
        self.sample_rate = cfg.sample_rate
        self.api_endpoint_base_url = cfg.endpoint_url
        self.model_version = cfg.model

    def run(self, audio_chunk_bytes):
        global interim_transcripts
        global audio_cursor
        global audio_cursor_l
        global segments_hypot
        global lock
        global predicted_transcript_hypot
        predicted_transcript_hypot = ""
        audio_cursor_l = []
        audio_cursor = 0
        interim_transcripts = []
        lock = threading.Lock()
        segments_hypot = {}

        def on_open(ws):
            def stream_audio(ws):
                global audio_cursor
                for chunk in audio_chunk_bytes:
                    ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                    time.sleep(self.chunk_size_ms / 1000)
                    audio_cursor += self.chunk_size_ms / 1000

                final_checkpoint = json.dumps({"checkpoint_id": "final"})
                ws.send(final_checkpoint, opcode=websocket.ABNF.OPCODE_TEXT)

            threading.Thread(target=stream_audio, args=(ws,)).start()

        def on_error(ws, error):
            print(f"Error: {error}")

        def on_message(ws, message):
            global predicted_transcript_hypot
            global interim_transcripts
            global audio_cursor
            global audio_cursor_l
            message = json.loads(message)
            if message.get("checkpoint_id") == "final":
                ws.close()
                return
            updated_segments = {segment["id"]: segment["text"] for segment in message["segments"]}
            audio_cursor_l.append(audio_cursor)
            with lock:
                segments_hypot.update(updated_segments)
                # clear_output(wait=True)
                logger.debug("\n" + "Transcription: " + "\n".join(f" - {k}: {v}" for k, v in segments_hypot.items()))
                predicted_transcript_hypot = " ".join(v for k, v in segments_hypot.items())
                interim_transcripts.append(predicted_transcript_hypot)

        params = urllib.parse.urlencode(
            {
                "language": "en",
            }
        )
        ws = websocket.WebSocketApp(
            f"{self.api_endpoint_base_url}?{params}",
            header={"Authorization": self.api_key, "model": self.model_version},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
        )

        ws.run_forever()

        return (
            predicted_transcript_hypot,
            interim_transcripts,
            audio_cursor_l,
            None,
            None,
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


class FireworksStreamingPipelineConfig(StreamingTranscriptionConfig):
    sample_rate: int
    channels: int
    sample_width: int
    chunksize_ms: float
    model: str


@register_pipeline
class FireworksStreamingPipeline(Pipeline):
    _config_class = FireworksStreamingPipelineConfig
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
        audio_data_byte = self.audio2chunks(y)
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
        pipeline = FireworksApi(self.config)
        return pipeline
