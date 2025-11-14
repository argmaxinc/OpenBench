# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from ...pipeline_prediction import Transcript
from ..base import PipelineConfig, PipelineOutput


class StreamingDiarizationConfig(PipelineConfig):
    endpoint_url: str


class StreamingDiarizationOutput(PipelineOutput[Transcript]): ...

