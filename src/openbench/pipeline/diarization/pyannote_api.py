# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import DiarizationSample
from ...engine import PyannoteAIApi, PyannoteApiDiarizationOutput
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["PyannoteApiPipeline", "PyannoteApiConfig"]

logger = get_logger(__name__)


class PyannoteApiConfig(DiarizationPipelineConfig):
    timeout: int = Field(
        default=1800,
        description="Timeout for the diarization job in seconds",
    )
    request_buffer: int = Field(
        default=30,
        description="Buffer for the request rate limit",
    )


TEMP_AUDIO_DIR = Path("audio_temp")


@register_pipeline
class PyannoteApiPipeline(Pipeline):
    _config_class = PyannoteApiConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(
        self,
    ) -> Callable[[dict[str, str | int | None]], PyannoteApiDiarizationOutput]:
        api = PyannoteAIApi(
            timeout=self.config.timeout,
            request_buffer=self.config.request_buffer,
            transcription=False,
        )
        return lambda input_sample: api(
            audio_path=input_sample["audio_path"],
            num_speakers=input_sample.get("num_speakers"),
        )

    def parse_input(self, input_sample: DiarizationSample) -> dict[str, str | int | None]:
        audio_path = input_sample.save_audio(TEMP_AUDIO_DIR)
        # setting as attribute to remove after parsing output
        self._audio_path = audio_path
        return {"audio_path": str(audio_path)}

    def parse_output(self, output: PyannoteApiDiarizationOutput) -> DiarizationOutput:
        result = DiarizationOutput(
            prediction=output.output.to_pyannote_annotation(),
        )
        # remove audio from temp
        self._audio_path.unlink()
        return result
