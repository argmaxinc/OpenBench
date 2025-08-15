# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Callable

import torch
from argmaxtools.utils import get_fastest_device
from pyannote.audio.sample import Annotation
from pydantic import model_validator

from ....dataset import DiarizationSample
from ....pipeline_prediction import DiarizationAnnotation
from ...base import Pipeline, PipelineType, register_pipeline
from ..common import DiarizationOutput, DiarizationPipelineConfig
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import (
    labels_to_pyannote_object,
)
from pathlib import Path
from pydantic import BaseModel

# Constants
TEMP_AUDIO_DIR = Path("./temp_audio")


__all__ = ["NeMoSortformerPipeline", "NeMoSortformerPipelineConfig"]


class NeMoSortformerPipelineInput(BaseModel):
    """Input for Sortformer pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    audio_path: Path
    keep_audio: bool = False
    annotation: DiarizationAnnotation | None = None


class NeMoSortformerPipelineConfig(DiarizationPipelineConfig):
    device: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    use_oracle_clustering: bool | None = None
    use_oracle_segmentation: bool | None = None
    use_float16: bool = False

    @model_validator(mode="after")
    def resolve_device(self) -> "NeMoSortformerPipelineConfig":
        self.device = get_fastest_device()
        return self


@register_pipeline
class NeMoSortformerPipeline(Pipeline):
    _config_class = NeMoSortformerPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(
        self,
    ) -> Callable[
        [dict[str, torch.FloatTensor | int]], DiarizationAnnotation
    ]:
        # load model from Hugging Face model card directly (You need a Hugging Face token)
        pipeline = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2",
            map_location="mps",
        )

        # switch to inference mode
        pipeline.eval()

        # VERY HIGH LATENCY
        CHUNK_SIZE = 340
        RIGHT_CONTEXT = 40
        FIFO_SIZE = 40
        UPDATE_PERIOD = 300
        SPEAKER_CACHE_SIZE = 188

        # # LOW LATENCY
        # CHUNK_SIZE = 6
        # RIGHT_CONTEXT = 7
        # FIFO_SIZE = 188
        # UPDATE_PERIOD = 144
        # SPEAKER_CACHE_SIZE = 188

        pipeline.sortformer_modules.chunk_len = CHUNK_SIZE
        pipeline.sortformer_modules.chunk_right_context = RIGHT_CONTEXT
        pipeline.sortformer_modules.fifo_len = FIFO_SIZE
        pipeline.sortformer_modules.spkcache_update_period = UPDATE_PERIOD
        pipeline.sortformer_modules.spkcache_len = SPEAKER_CACHE_SIZE

        def call_pipeline(
            inputs: NeMoSortformerPipelineInput,
        ) -> DiarizationAnnotation:
            with torch.autocast(
                device_type=self.config.device,
                enabled=True,
                dtype=(
                    torch.float16
                    if self.config.use_float16
                    else torch.float32
                ),
            ):
                result = pipeline.diarize(str(inputs.audio_path), batch_size=1)
            annot: Annotation = labels_to_pyannote_object(result[0])
            return DiarizationAnnotation.from_pyannote_annotation(annot)

        return call_pipeline

    def parse_input(
        self, input_sample: DiarizationSample
    ) -> dict[str, torch.FloatTensor | int]:
        parsed_input = NeMoSortformerPipelineInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
        )
        if (
            self.config.use_oracle_clustering
            or self.config.use_oracle_segmentation
        ):
            parsed_input.annotation = input_sample.annotation
        return parsed_input

    def parse_output(self, output: DiarizationAnnotation) -> DiarizationOutput:
        return DiarizationOutput(prediction=output)
