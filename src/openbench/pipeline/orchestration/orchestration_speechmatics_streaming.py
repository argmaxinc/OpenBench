# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import numpy as np
from pydantic import Field

from ...dataset import OrchestrationSample
from ...pipeline import Pipeline, PipelineConfig, register_pipeline
from ...pipeline_prediction import Transcript, Word
from ...types import PipelineType
from ..streaming_transcription.speechmatics import SpeechmaticsApi
from .common import OrchestrationOutput


class SpeechmaticsStreamingOrchestrationPipelineConfig(PipelineConfig):
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
    enable_diarization: bool = Field(
        default=True,
        description="Whether to enable speaker diarization"
    )


@register_pipeline
class SpeechmaticsStreamingOrchestrationPipeline(Pipeline):
    _config_class = SpeechmaticsStreamingOrchestrationPipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self):
        """Build Speechmatics streaming API with diarization."""
        # Create a modified config for the streaming API
        from types import SimpleNamespace

        api_config = SimpleNamespace(
            sample_rate=self.config.sample_rate,
            language=self.config.language,
            operating_point=self.config.operating_point,
            max_delay=self.config.max_delay,
            enable_partials=self.config.enable_partials,
            enable_diarization=self.config.enable_diarization,
        )

        pipeline = SpeechmaticsApi(api_config)
        return pipeline

    def parse_input(self, input_sample: OrchestrationSample):
        """Convert audio waveform to bytes for streaming."""
        y = input_sample.waveform
        y_int16 = (y * 32767).astype(np.int16)
        audio_data_byte = y_int16.tobytes()
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
            transcript_words = output.get("transcript", "").split()
            timestamp_idx = 0

            for timestamp_group in output["model_timestamps_confirmed"]:
                for word_info in timestamp_group:
                    if timestamp_idx < len(transcript_words):
                        words.append(Word(
                            word=transcript_words[timestamp_idx],
                            start=word_info.get("start"),
                            end=word_info.get("end"),
                            speaker=None,
                        ))
                        timestamp_idx += 1

        # Create final transcript with speaker-attributed words
        transcript = Transcript(words=words)

        return OrchestrationOutput(
            prediction=transcript,
            transcription_output=None,
            diarization_output=None,
        )

