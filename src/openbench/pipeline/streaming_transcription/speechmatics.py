# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import io
import os

import numpy as np
import speechmatics
from argmaxtools.utils import get_logger
from pydantic import Field
from speechmatics.models import ServerMessageType

from openbench.dataset import StreamingSample

from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import StreamingTranscript
from ...types import PipelineType
from .common import StreamingTranscriptionConfig, StreamingTranscriptionOutput


logger = get_logger(__name__)


class SpeechmaticsApi:
    def __init__(self, cfg) -> None:
        self.api_key = os.getenv("SPEECHMATICS_API_KEY")
        assert (
            self.api_key is not None
        ), "Please set SPEECHMATICS_API_KEY in environment"
        self.language = getattr(cfg, 'language', 'en')
        self.operating_point = getattr(cfg, 'operating_point', 'enhanced')
        self.max_delay = getattr(cfg, 'max_delay', 1)
        self.enable_partials = getattr(cfg, 'enable_partials', True)
        self.sample_rate = cfg.sample_rate
        self.connection_url = os.getenv(
            "SPEECHMATICS_URL", "wss://eu2.rt.speechmatics.com/v2"
        )
        self.enable_diarization = getattr(
            cfg, 'enable_diarization', False
        )

    def __call__(self, sample):
        # Sample must be in bytes (raw audio data)
        transcript = ""
        interim_transcripts = []
        audio_cursor_l = []
        confirmed_interim_transcripts = []
        confirmed_audio_cursor_l = []
        model_timestamps_hypothesis = []
        model_timestamps_confirmed = []
        words_with_speakers = []

        # Create audio cursor tracker
        audio_cursor = [0.0]

        # Create a transcription client
        ws = speechmatics.client.WebsocketClient(
            speechmatics.models.ConnectionSettings(
                url=self.connection_url,
                auth_token=self.api_key,
            )
        )

        # Define event handler for partial transcripts
        def handle_partial_transcript(msg):
            nonlocal interim_transcripts, audio_cursor_l
            nonlocal model_timestamps_hypothesis

            metadata = msg.get('metadata', {})
            partial_transcript = metadata.get('transcript', '')

            if partial_transcript:
                audio_cursor_l.append(audio_cursor[0])
                interim_transcripts.append(
                    transcript + " " + partial_transcript
                )

                # Collect word timestamps if available
                results = msg.get('results', [])
                if results:
                    words = []
                    for result in results:
                        if result.get('type') == 'word':
                            words.append({
                                'start': result.get('start_time', 0),
                                'end': result.get('end_time', 0),
                            })
                    if words:
                        model_timestamps_hypothesis.append(words)

                logger.debug(f"[partial] {partial_transcript}")

        # Define event handler for full transcripts
        def handle_transcript(msg):
            nonlocal transcript, confirmed_interim_transcripts
            nonlocal confirmed_audio_cursor_l
            nonlocal model_timestamps_confirmed, words_with_speakers

            metadata = msg.get('metadata', {})
            full_transcript = metadata.get('transcript', '')

            if full_transcript:
                confirmed_audio_cursor_l.append(audio_cursor[0])
                transcript = transcript + " " + full_transcript
                confirmed_interim_transcripts.append(transcript)

                # Collect word timestamps and speaker info
                results = msg.get('results', [])
                if results:
                    words = []
                    for result in results:
                        if result.get('type') == 'word':
                            # Get alternatives array
                            alternatives = result.get('alternatives', [])
                            if alternatives:
                                # Take first alternative
                                alternative = alternatives[0]

                                word_data = {
                                    'start': result.get('start_time', 0),
                                    'end': result.get('end_time', 0),
                                }
                                words.append(word_data)

                                # Collect speaker info if diarization
                                if self.enable_diarization:
                                    speaker_info = alternative.get(
                                        'speaker', None
                                    )
                                    word_content = alternative.get(
                                        'content', ''
                                    )
                                    if speaker_info is not None:
                                        words_with_speakers.append({
                                            'word': word_content,
                                            'speaker': (
                                                f"SPEAKER_{speaker_info}"
                                            ),
                                            'start': result.get(
                                                'start_time', 0
                                            ),
                                            'end': result.get(
                                                'end_time', 0
                                            ),
                                        })

                    if words:
                        model_timestamps_confirmed.append(words)

                logger.debug(f"[FULL] {full_transcript}")

        # Register event handlers
        ws.add_event_handler(
            event_name=ServerMessageType.AddPartialTranscript,
            event_handler=handle_partial_transcript,
        )

        ws.add_event_handler(
            event_name=ServerMessageType.AddTranscript,
            event_handler=handle_transcript,
        )

        # Audio settings
        settings = speechmatics.models.AudioSettings(
            sample_rate=self.sample_rate,
            encoding='pcm_s16le',
        )

        # Transcription config
        conf_dict = {
            'operating_point': self.operating_point,
            'language': self.language,
            'enable_partials': self.enable_partials,
            'max_delay': self.max_delay,
        }

        # Enable diarization if requested
        if self.enable_diarization:
            conf_dict['diarization'] = 'speaker'

        conf = speechmatics.models.TranscriptionConfig(**conf_dict)

        # Create a BytesIO stream from the audio data
        audio_stream = io.BytesIO(sample)

        try:
            # Run transcription synchronously
            ws.run_synchronously(audio_stream, conf, settings)
        except Exception as e:
            logger.error(f"Speechmatics transcription error: {e}")
            raise

        return {
            "transcript": transcript.strip(),
            "interim_transcripts": interim_transcripts,
            "audio_cursor": audio_cursor_l,
            "confirmed_interim_transcripts": (
                confirmed_interim_transcripts
            ),
            "confirmed_audio_cursor": confirmed_audio_cursor_l,
            "model_timestamps_hypothesis": (
                model_timestamps_hypothesis
            ),
            "model_timestamps_confirmed": (
                model_timestamps_confirmed
            ),
            "words_with_speakers": words_with_speakers,
        }


class SpeechmaticsStreamingPipelineConfig(StreamingTranscriptionConfig):
    sample_rate: int = Field(
        default=16000,
        description="Sample rate of the audio"
    )
    language: str = Field(
        default="en",
        description="Language code for transcription"
    )
    operating_point: str = Field(
        default="enhanced",
        description="Operating point (standard or enhanced)"
    )
    max_delay: int = Field(
        default=1,
        description="Maximum delay in seconds"
    )
    enable_partials: bool = Field(
        default=True,
        description="Enable partial transcripts"
    )


@register_pipeline
class SpeechmaticsStreamingPipeline(Pipeline):
    _config_class = SpeechmaticsStreamingPipelineConfig
    pipeline_type = PipelineType.STREAMING_TRANSCRIPTION

    def parse_input(self, input_sample: StreamingSample):
        y = input_sample.waveform
        y_int16 = (y * 32767).astype(np.int16)
        audio_data_byte = y_int16.tobytes()
        return audio_data_byte

    def parse_output(
        self, output
    ) -> StreamingTranscriptionOutput:
        model_timestamps_hypothesis = (
            output["model_timestamps_hypothesis"]
        )
        model_timestamps_confirmed = (
            output["model_timestamps_confirmed"]
        )

        prediction = StreamingTranscript(
            transcript=output["transcript"],
            audio_cursor=output["audio_cursor"],
            interim_results=output["interim_transcripts"],
            confirmed_audio_cursor=output["confirmed_audio_cursor"],
            confirmed_interim_results=(
                output["confirmed_interim_transcripts"]
            ),
            model_timestamps_hypothesis=model_timestamps_hypothesis,
            model_timestamps_confirmed=model_timestamps_confirmed,
        )

        return StreamingTranscriptionOutput(prediction=prediction)

    def build_pipeline(self):
        pipeline = SpeechmaticsApi(self.config)
        return pipeline
