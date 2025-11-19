# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import numpy as np
from pydantic import Field

from ...dataset import OrchestrationSample
from ...pipeline import Pipeline, PipelineConfig, register_pipeline
from ...pipeline_prediction import Transcript, Word
from ...types import PipelineType
from ..streaming_transcription.deepgram import DeepgramApi
from .common import OrchestrationOutput


class DeepgramStreamingOrchestrationPipelineConfig(PipelineConfig):
    sample_rate: int = Field(
        default=16000,
        description="Sample rate of the audio"
    )
    channels: int = Field(
        default=1,
        description="Number of audio channels"
    )
    sample_width: int = Field(
        default=2,
        description="Sample width in bytes"
    )
    realtime_resolution: float = Field(
        default=0.020,
        description="Real-time resolution for streaming"
    )
    model_version: str = Field(
        default="nova-3",
        description=(
            "The model to use for real-time transcription "
            "with diarization"
        )
    )
    enable_diarization: bool = Field(
        default=True,
        description="Whether to enable speaker diarization"
    )


@register_pipeline
class DeepgramStreamingOrchestrationPipeline(Pipeline):
    _config_class = DeepgramStreamingOrchestrationPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self):
        """Build Deepgram streaming API with diarization enabled."""
        # Create a modified config for the streaming API
        from types import SimpleNamespace

        api_config = SimpleNamespace(
            channels=self.config.channels,
            sample_width=self.config.sample_width,
            sample_rate=self.config.sample_rate,
            realtime_resolution=self.config.realtime_resolution,
            model_version=self.config.model_version,
            enable_diarization=self.config.enable_diarization,
        )

        pipeline = DeepgramApi(api_config)
        return pipeline

    def parse_input(self, input_sample: OrchestrationSample):
        """Convert audio waveform to bytes for streaming."""
        y = input_sample.waveform
        y_int16 = (y * 32767).astype(np.int16)
        audio_data_byte = y_int16.T.tobytes()
        return audio_data_byte

    def parse_output(self, output) -> OrchestrationOutput:
        """Parse output to extract transcription and diarization."""
        # Extract words with speaker info if diarization enabled
        words = []

        if (
            "words_with_speakers" in output and
            output["words_with_speakers"]
        ):
            # This comes from diarization-enabled streaming
            for word_info in output["words_with_speakers"]:
                words.append(Word(
                    word=word_info.get("word", ""),
                    start=word_info.get("start"),
                    end=word_info.get("end"),
                    speaker=word_info.get("speaker"),
                ))
        elif (
            "model_timestamps_confirmed" in output and
            output["model_timestamps_confirmed"]
        ):
            # Fallback to regular transcription without speaker
            for timestamp_group in output["model_timestamps_confirmed"]:
                for word_info in timestamp_group:
                    if "word" in word_info:
                        words.append(Word(
                            word=word_info.get("word", ""),
                            start=word_info.get("start"),
                            end=word_info.get("end"),
                            speaker=None,
                        ))

        # Create final transcript with speaker-attributed words
        transcript = Transcript(words=words)

        return OrchestrationOutput(
            prediction=transcript,
            transcription_output=None,
            diarization_output=None,
        )
