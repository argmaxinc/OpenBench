import json
import os
import threading
import time
from urllib.parse import urlencode

import torch
import torchaudio
import websocket
from argmaxtools.utils import get_logger

from ...dataset import StreamingSample
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import StreamingTranscript
from ...types import PipelineType
from .common import StreamingTranscriptionConfig, StreamingTranscriptionOutput


logger = get_logger(__name__)


class AssemblyAIApi:
    def __init__(self, cfg) -> None:
        self.chunk_size_ms = cfg.chunksize_ms
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        assert self.api_key is not None, "Please set ASSEMBLYAI_API_KEY in environment"
        self.channels = cfg.channels
        self.sample_width = cfg.sample_width
        self.sample_rate = cfg.sample_rate
        CONNECTION_PARAMS = {
            "sample_rate": self.sample_rate,
            "format_turns": False,
        }
        self.api_endpoint_base_url = cfg.endpoint_url
        self.api_endpoint = f"{self.api_endpoint_base_url}?{urlencode(CONNECTION_PARAMS)}"

    def scale_model_timestamps(self, timestamps):
        for sublist in timestamps:
            for item in sublist:
                item["start"] /= 1000
                item["end"] /= 1000
        return timestamps

    def run(self, data):
        global interim_transcripts
        global audio_cursor
        global audio_cursor_l
        global segments_hypot
        global lock
        global predicted_transcript_hypot
        global confirmed_interim_transcripts
        global model_timestamps_hypot
        global model_timestamps_confirmed
        audio_cursor_l = []
        audio_cursor = 0
        interim_transcripts = []
        confirmed_interim_transcripts = []
        model_timestamps_confirmed = []
        model_timestamps_hypot = []
        lock = threading.Lock()
        segments_hypot = {}
        segments = {}

        def on_open(ws):
            def stream_audio(ws):
                global audio_cursor
                for chunk in data:
                    ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                    time.sleep(self.chunk_size_ms / 1000)
                    audio_cursor += self.chunk_size_ms / 1000

                final_checkpoint = json.dumps({"type": "Terminate"})
                ws.send(final_checkpoint, opcode=websocket.ABNF.OPCODE_TEXT)

            threading.Thread(target=stream_audio, args=(ws,)).start()

        def on_error(ws, error):
            print(f"Error: {error}")

        def on_message(ws, message):
            global predicted_transcript_hypot
            global audio_cursor
            global audio_cursor_l
            global interim_transcripts
            global confirmed_interim_transcripts
            global model_timestamps_hypot
            global model_timestamps_confirmed
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "Turn":
                    if data["transcript"] != "":
                        audio_cursor_l.append(audio_cursor)
                        model_timestamps_confirmed.append(
                            [item for item in data["words"] if item.get("word_is_final", True)]
                        )
                        model_timestamps_hypot.append(data["words"])
                        updated_segments_hypot = {data["turn_order"]: " ".join(word["text"] for word in data["words"])}
                        updated_segments = {data["turn_order"]: data["transcript"]}

                        with lock:
                            segments.update(updated_segments)
                            confirmed_predicted_transcript = " ".join(v for k, v in segments.items())
                            confirmed_interim_transcripts.append(confirmed_predicted_transcript)

                        with lock:
                            segments_hypot.update(updated_segments_hypot)
                            predicted_transcript_hypot = " ".join(v for k, v in segments_hypot.items())
                            logger.debug("\n" + "Transcription: " + predicted_transcript_hypot)
                            interim_transcripts.append(predicted_transcript_hypot)

                elif msg_type == "Termination":
                    audio_duration = data.get("audio_duration_seconds", 0)
                    session_duration = data.get("session_duration_seconds", 0)
                    print(
                        f"Session Terminated: Audio Duration={audio_duration}s, Session Duration={session_duration}s"
                    )
            except json.JSONDecodeError as e:
                print(f"Error decoding message: {e}")
            except Exception as e:
                print(f"Error handling message: {e}")

        ws = websocket.WebSocketApp(
            self.api_endpoint,
            header={"Authorization": self.api_key},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
        )

        ws.run_forever()

        return (
            predicted_transcript_hypot,
            interim_transcripts,
            audio_cursor_l,
            confirmed_interim_transcripts,
            audio_cursor_l,
            model_timestamps_hypot,
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
            model_timestamps_hypot,
            model_timestamps_confirmed,
        ) = self.run(sample)
        model_timestamps_hypot = self.scale_model_timestamps(model_timestamps_hypot)
        # TODO: scaling model_timestamps_hypot also scales model_timestamps_confirmed
        # consider decoupling shared variables
        # model_timestamps_confirmed = self.scale_model_timestamps(model_timestamps_confirmed)
        return {
            "transcript": transcript,
            "interim_transcripts": interim_transcripts,
            "audio_cursor": audio_cursor_l,
            "confirmed_interim_transcripts": confirmed_interim_transcripts,
            "confirmed_audio_cursor": confirmed_audio_cursor_l,
            "model_timestamps_hypot": model_timestamps_hypot,
            "model_timestamps_confirmed": model_timestamps_confirmed,
        }


class AssemblyAIStreamingPipelineConfig(StreamingTranscriptionConfig):
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    chunksize_ms: float = 50
    endpoint_url: str = "wss://streaming.assemblyai.com/v3/ws"


@register_pipeline
class AssemblyAIStreamingPipeline(Pipeline):
    _config_class = AssemblyAIStreamingPipelineConfig
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
        model_timestamps_hypothesis = output["model_timestamps_hypot"]
        model_timestamps_confirmed = output["model_timestamps_confirmed"]

        # Filter to only include start and end timestamps, removing text/word fields
        if model_timestamps_hypothesis is not None:
            model_timestamps_hypothesis = [
                [
                    {"start": word["start"], "end": word["end"]}
                    for word in interim_result_words
                    if "start" in word and "end" in word
                ]
                for interim_result_words in model_timestamps_hypothesis
            ]

        if model_timestamps_confirmed is not None:
            model_timestamps_confirmed = [
                [
                    {"start": word["start"], "end": word["end"]}
                    for word in interim_result_words
                    if "start" in word and "end" in word
                ]
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
        pipeline = AssemblyAIApi(self.config)
        return pipeline
