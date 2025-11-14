# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Shared types used across the openbench package."""

from enum import Enum
from typing import Protocol, runtime_checkable


class PipelineType(Enum):
    DIARIZATION = "diarization"
    TRANSCRIPTION = "transcription"
    ORCHESTRATION = "orchestration"
    STREAMING_TRANSCRIPTION = "streaming_transcription"
    STREAMING_DIARIZATION = "streaming_diarization"


# All prediction classes that we output should conform to this
# TODO: add `load_from_annotation` to protocol
@runtime_checkable
class PredictionProtocol(Protocol):
    def to_annotation_file(self, output_dir: str, filename: str) -> str:
        """
        Must implement a method to save the prediction to a file.

        Args:
            output_dir: The directory to save the prediction to.
            filename: The filename to save the prediction to without the extension

        Returns:
            The path to the saved prediction.
        """
        pass
