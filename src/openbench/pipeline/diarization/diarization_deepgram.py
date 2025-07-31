from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from deepgram import PrerecordedOptions
from pyannote.core import Segment
from pydantic import Field

from ...dataset import DiarizationSample
from ...engine import DeepgramApi, DeepgramApiResponse
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["DeepgramDiarizationPipeline", "DeepgramDiarizationPipelineConfig"]

TEMP_AUDIO_DIR = Path("audio_temp")

logger = get_logger(__name__)


class DeepgramDiarizationPipelineConfig(DiarizationPipelineConfig):
    model_version: str = Field(
        default="nova-3",
        description="The version of the Deepgram model to use",
    )


@register_pipeline
class DeepgramDiarizationPipeline(Pipeline):
    _config_class = DeepgramDiarizationPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[Path], DeepgramApiResponse]:
        options = PrerecordedOptions(
            model=self.config.model_version, diarize=True, smart_format=True, detect_language=True
        )

        if self.config.use_exact_num_speakers:
            logger.warning("`use_exact_num_speakers` is not supported for DeepgramDiarizationPipeline")

        self.api_client = DeepgramApi(options)

        def transcribe(audio_path: Path) -> DeepgramApiResponse:
            response = self.api_client.transcribe(audio_path)
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def parse_input(self, input_sample: DiarizationSample) -> Path:
        return input_sample.save_audio(output_dir=TEMP_AUDIO_DIR)

    def parse_output(self, output: DeepgramApiResponse) -> DiarizationOutput:
        annotation = DiarizationAnnotation()
        for word, speaker, start, end in zip(output.words, output.speakers, output.start, output.end):
            annotation[Segment(start, end)] = f"SPEAKER_{speaker}"

        return DiarizationOutput(prediction=annotation)
