import json
import os
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import torch
import torchaudio
import websocket
from argmaxtools.utils import get_logger

from ...dataset import TranscriptionSample
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput

logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


class AssemblyAIApi:
    def __init__(self, cfg) -> None:
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        assert self.api_key is not None, "Please set ASSEMBLYAI_API_KEY in environment"
        
        self.sample_rate = cfg.sample_rate
        self.channels = cfg.channels
        self.sample_width = cfg.sample_width
        self.chunksize_ms = cfg.chunksize_ms
        self.api_endpoint_base_url = cfg.endpoint_url

    def get_api_endpoint(self, keywords=None):
        """Build API endpoint with optional keywords."""
        CONNECTION_PARAMS = {
            "sample_rate": self.sample_rate,
            "format_turns": True,  # Request formatted final transcripts
        }

        # Add keywords if provided
        if keywords:
            CONNECTION_PARAMS["keyterms_prompt"] = json.dumps(keywords)

        return f"{self.api_endpoint_base_url}?{urlencode(CONNECTION_PARAMS)}"

    def run(self, audio_chunks, keywords=None):
        """Run transcription on audio chunks."""
        transcript_parts = []
        session_complete = threading.Event()
        error_occurred = threading.Event()
        error_message = None
        
        # Get API endpoint with keywords
        api_endpoint = self.get_api_endpoint(keywords)

        def on_open(ws):
            def stream_audio(ws):
                try:
                    for chunk in audio_chunks:
                        if error_occurred.is_set():
                            break
                        ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                        time.sleep(self.chunksize_ms / 1000)
                    
                    # Send termination message
                    terminate_message = json.dumps({"type": "Terminate"})
                    ws.send(terminate_message, opcode=websocket.ABNF.OPCODE_TEXT)
                except Exception as e:
                    logger.error(f"Error streaming audio: {e}")
                    nonlocal error_message
                    error_message = str(e)
                    error_occurred.set()

            threading.Thread(target=stream_audio, args=(ws,)).start()

        def on_message(ws, message):
            nonlocal error_message
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "Begin":
                    session_id = data.get('id')
                    logger.debug(f"Session began: ID={session_id}")
                    
                elif msg_type == "Turn":
                    transcript = data.get('transcript', '')
                    formatted = data.get('turn_is_formatted', False)
                    
                    if formatted and transcript.strip():
                        transcript_parts.append(transcript.strip())
                        logger.debug(f"Received transcript: {transcript}")
                        
                elif msg_type == "Termination":
                    audio_duration = data.get('audio_duration_seconds', 0)
                    session_duration = data.get('session_duration_seconds', 0)
                    logger.debug(f"Session terminated: Audio={audio_duration}s, Session={session_duration}s")
                    session_complete.set()
                    
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding message: {e}")
                error_message = f"JSON decode error: {e}"
                error_occurred.set()
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                error_message = f"Message handling error: {e}"
                error_occurred.set()

        def on_error(ws, error):
            nonlocal error_message
            logger.error(f"WebSocket error: {error}")
            error_message = f"WebSocket error: {error}"
            error_occurred.set()

        def on_close(ws, close_status_code, close_msg):
            logger.debug(f"WebSocket closed: Status={close_status_code}, Msg={close_msg}")
            session_complete.set()

        # Create and run WebSocket connection
        ws = websocket.WebSocketApp(
            api_endpoint,
            header={"Authorization": self.api_key},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # Run WebSocket in a separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        # Wait for completion or error
        while ws_thread.is_alive():
            if error_occurred.wait(timeout=0.1):
                ws.close()
                break
            if session_complete.wait(timeout=0.1):
                break

        ws_thread.join(timeout=5.0)

        if error_occurred.is_set():
            raise RuntimeError(f"AssemblyAI transcription failed: {error_message}")

        # Combine all transcript parts
        full_transcript = " ".join(transcript_parts).strip()
        return full_transcript

    def audio_to_chunks(self, audio_data, sample_rate):
        """Convert audio data to chunks suitable for streaming."""
        # Ensure audio is tensor
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data)
            
        # Add batch dimension if needed
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)

        # Resample to target sample rate
        if sample_rate != self.sample_rate:
            audio_data = torchaudio.functional.resample(
                audio_data, sample_rate, self.sample_rate
            )

        # Convert to mono if needed
        if audio_data.shape[0] > 1:
            audio_data = torch.mean(audio_data, dim=0, keepdim=True)

        # Split into chunks
        chunk_size = int(self.chunksize_ms * self.sample_rate / 1000)
        audio_chunks = torch.split(audio_data, chunk_size, dim=1)

        # Convert chunks to bytes
        chunk_bytes = []
        for chunk in audio_chunks:
            # Convert to int16 and then to bytes
            chunk_int16 = (chunk * 32768.0).clamp(-32768, 32767).to(torch.int16)
            chunk_bytes.append(chunk_int16.numpy().tobytes())

        return chunk_bytes

    def __call__(self, sample, keywords=None):
        """Process a transcription sample."""
        audio_chunks = self.audio_to_chunks(sample.waveform, sample.sample_rate)
        transcript = self.run(audio_chunks, keywords)
        return transcript


class AssemblyAITranscriptionPipelineConfig(TranscriptionConfig):
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    chunksize_ms: float = 50
    endpoint_url: str = "wss://streaming.assemblyai.com/v3/ws"


@register_pipeline
class AssemblyAITranscriptionPipeline(Pipeline):
    _config_class = AssemblyAITranscriptionPipelineConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[TranscriptionSample], str]:
        assemblyai_api = AssemblyAIApi(self.config)

        def transcribe(sample: TranscriptionSample) -> str:
            return assemblyai_api(sample, keywords=self.current_keywords)

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> TranscriptionSample:
        """Override to extract keywords from sample before processing."""
        self.current_keywords = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get('dictionary', [])
            if keywords:
                self.current_keywords = keywords

        return input_sample

    def parse_output(self, output: str) -> TranscriptionOutput:
        # Split transcript into words
        words = output.split() if output else []
        transcript = Transcript.from_words_info(words=words)
        return TranscriptionOutput(prediction=transcript)
