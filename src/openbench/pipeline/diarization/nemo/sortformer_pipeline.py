# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path
from typing import Callable

import torch
from argmaxtools.utils import get_fastest_device
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import (
    labels_to_pyannote_object,
)
from pyannote.audio.sample import Annotation
from pydantic import BaseModel, Field

from ....dataset import DiarizationSample
from ....pipeline_prediction import DiarizationAnnotation
from ...base import Pipeline, PipelineType, register_pipeline
from ..common import DiarizationOutput, DiarizationPipelineConfig


# Constants
TEMP_AUDIO_DIR = Path("./temp_audio")
SORTFORMER_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2"


__all__ = ["NeMoSortformerPipeline", "NeMoSortformerPipelineConfig"]


class NeMoSortformerPipelineInput(BaseModel):
    """Input for Sortformer pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    audio_path: Path
    keep_audio: bool = False


class NeMoSortformerPipelineConfig(DiarizationPipelineConfig):
    device: str = Field(
        default_factory=get_fastest_device,
        description="PyTorch device where Sortformer will run its inference.",
    )
    use_float16: bool = Field(
        default=True,
        description="Whether to use float16 for the Sortformer model.",
    )
    chunk_size: int = Field(
        default=340,
        description="Chunk size for the Sortformer model.",
    )
    right_context: int = Field(
        default=40,
        description="Right context for the Sortformer model.",
    )
    fifo_size: int = Field(
        default=40,
        description="FIFO size for the Sortformer model.",
    )
    update_period: int = Field(
        default=300,
        description="Update period for the Sortformer model.",
    )
    speaker_cache_size: int = Field(
        default=188,
        description="Speaker cache size for the Sortformer model.",
    )


@register_pipeline
class NeMoSortformerPipeline(Pipeline):
    _config_class = NeMoSortformerPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(
        self,
    ) -> Callable[[dict[str, torch.FloatTensor | int]], DiarizationAnnotation]:
        # load model from Hugging Face model card directly (You need a Hugging Face token)
        pipeline = SortformerEncLabelModel.from_pretrained(
            SORTFORMER_MODEL_ID,
            map_location=self.config.device,
        )

        # switch to inference mode
        pipeline.eval()

        pipeline.sortformer_modules.chunk_len = self.config.chunk_size
        pipeline.sortformer_modules.chunk_right_context = self.config.right_context
        pipeline.sortformer_modules.fifo_len = self.config.fifo_size
        pipeline.sortformer_modules.spkcache_update_period = self.config.update_period
        pipeline.sortformer_modules.spkcache_len = self.config.speaker_cache_size

        def call_pipeline(
            inputs: NeMoSortformerPipelineInput,
        ) -> DiarizationAnnotation:
            with torch.autocast(
                device_type=self.config.device,
                enabled=True,
                dtype=(torch.float16 if self.config.use_float16 else torch.float32),
            ):
                result = pipeline.diarize(str(inputs.audio_path), batch_size=1)
            annot: Annotation = labels_to_pyannote_object(result[0])
            # Delete temp audio file
            inputs.audio_path.unlink()
            return DiarizationAnnotation.from_pyannote_annotation(annot)

        return call_pipeline

    def parse_input(self, input_sample: DiarizationSample) -> dict[str, torch.FloatTensor | int]:
        parsed_input = NeMoSortformerPipelineInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
        )
        return parsed_input

    def parse_output(self, output: DiarizationAnnotation) -> DiarizationOutput:
        return DiarizationOutput(prediction=output)
